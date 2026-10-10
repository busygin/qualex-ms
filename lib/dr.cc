/***********************************************************************
!! dr.cc: Douglas-Rachford iterations between a sphere and a set given !!
!! by its projection, and on further sets (proposed by S. Busygin).    !!
***********************************************************************/

#include <math.h>
#include <string.h>
#include <memory>
#include <cblas.h>

#include "dr.h"

using namespace std;

void clip_negative(double* y, int n) {
  for(int i=0;i<n;++i) if(y[i]<0.0) y[i] = 0.0;
}

int douglas_rachford(const Sphere& s, const Projection& project_c,
                     int nb, double* v, double* e, vector<bool>& done,
                     int iters, double tol, long* steps) {
  int n = s.n, k = s.k;
  long work = 0;
  vector<double> d((size_t)n*nb);  // v - center, then its projection on span Q
  vector<double> p((size_t)k*nb);  // the coordinates of that projection
  vector<double> y(n);             // 2e - v, then P_C of it; P_C(e) for the test
  done.assign(nb,false);
  for(int it=0;it<iters;++it) {
    for(int t=0;t<nb;++t)
      for(int i=0;i<n;++i) d[(size_t)t*n+i] = v[(size_t)t*n+i]-s.center[i];
    cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,k,nb,n,
                1.0,s.q,n,d.data(),n,0.0,p.data(),k);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,nb,k,
                1.0,s.q,n,p.data(),k,0.0,d.data(),n);
    int live = 0;
    for(int t=0;t<nb;++t) {
      if(done[t]) continue;
      ++work;
      double* dt = &d[(size_t)t*n];
      double* et = e+(size_t)t*n;
      double* vt = v+(size_t)t*n;
      double nu = 0.0;
      for(int i=0;i<n;++i) nu += dt[i]*dt[i];
      if(!(nu>0.0)) {  // v - center is orthogonal to the sphere: take the center
        memcpy(et,s.center,sizeof(double)*n);
        done[t] = true;
        continue;
      }
      nu = s.radius/sqrt(nu);
      double all = 0.0;
      for(int i=0;i<n;++i) {
        double ei = et[i] = s.center[i]+nu*dt[i];
        all += ei*ei;
      }
      memcpy(y.data(),et,sizeof(double)*n);
      project_c(y.data(),n);
      double off = 0.0;  // |P_C(e) - e|^2
      for(int i=0;i<n;++i) off += (y[i]-et[i])*(y[i]-et[i]);
      if(off<=tol*tol*all) { done[t] = true; continue; }
      for(int i=0;i<n;++i) y[i] = 2.0*et[i]-vt[i];
      project_c(y.data(),n);
      for(int i=0;i<n;++i) vt[i] += y[i]-et[i];
      ++live;
    }
    if(!live) break;
  }
  if(steps) *steps += work;
  int n_done = 0;
  for(int t=0;t<nb;++t) if(done[t]) ++n_done;
  return n_done;
}

BlockProjection columnwise(const Projection& project) {
  return [project](double* y, int n, int nb) {
    for(int t=0;t<nb;++t) project(y+(size_t)t*n,n);
  };
}

BlockProjection sphere_projection(const Sphere& s) {
  shared_ptr<vector<double>> d = make_shared<vector<double>>();  // y - center, projected on span Q
  shared_ptr<vector<double>> p = make_shared<vector<double>>();  // its coordinates
  return [s,d,p](double* y, int n, int nb) {
    d->resize((size_t)n*nb);
    p->resize((size_t)s.k*nb);
    double* dd = d->data();
    for(int t=0;t<nb;++t)
      for(int i=0;i<n;++i) dd[(size_t)t*n+i] = y[(size_t)t*n+i]-s.center[i];
    cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,s.k,nb,n,
                1.0,s.q,n,dd,n,0.0,p->data(),s.k);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,nb,s.k,
                1.0,s.q,n,p->data(),s.k,0.0,dd,n);
    for(int t=0;t<nb;++t) {
      double* dt = dd+(size_t)t*n;
      double* yt = y+(size_t)t*n;
      double nu = 0.0;
      for(int i=0;i<n;++i) nu += dt[i]*dt[i];
      if(nu>0.0) {
        nu = s.radius/sqrt(nu);
        for(int i=0;i<n;++i) yt[i] = s.center[i]+nu*dt[i];
      } else
        for(int i=0;i<n;++i) yt[i] = s.center[i]+s.radius*s.q[i];
    }
  };
}

int douglas_rachford_concur(const vector<BlockProjection>& sets,
                            int n, int nb, double* v, double* e,
                            vector<bool>& done, int iters, double tol,
                            long* steps) {
  int K = (int)sets.size();
  long work = 0;
  size_t block = (size_t)n*nb;
  vector<double> p((size_t)K*block);  // 2e - v_k, then P_k of it
  done.assign(nb,false);
  for(int it=0;it<iters;++it) {
    for(size_t i=0;i<block;++i) {
      double sum = 0.0;
      for(int k=0;k<K;++k) sum += v[k*block+i];
      e[i] = sum/K;
    }
    for(int k=0;k<K;++k) {
      double* pk = &p[k*block];
      for(size_t i=0;i<block;++i) pk[i] = 2.0*e[i]-v[k*block+i];
      sets[k](pk,n,nb);
    }
    int live = 0;
    for(int t=0;t<nb;++t) {
      if(done[t]) continue;
      ++work;
      const double* et = e+(size_t)t*n;
      double all = 0.0, off = 0.0;
      for(int i=0;i<n;++i) all += et[i]*et[i];
      for(int k=0;k<K;++k) {
        const double* pkt = &p[k*block+(size_t)t*n];
        double d2 = 0.0;
        for(int i=0;i<n;++i) d2 += (pkt[i]-et[i])*(pkt[i]-et[i]);
        if(d2>off) off = d2;
      }
      if(off<=tol*tol*all) {
        done[t] = true;
        continue;
      }
      for(int k=0;k<K;++k) {
        double* vkt = v+k*block+(size_t)t*n;
        const double* pkt = &p[k*block+(size_t)t*n];
        for(int i=0;i<n;++i) vkt[i] += pkt[i]-et[i];
      }
      ++live;
    }
    if(!live) break;
  }
  if(steps) *steps += work;
  int n_done = 0;
  for(int t=0;t<nb;++t) if(done[t]) ++n_done;
  return n_done;
}

int douglas_rachford_product(const Sphere& s, const vector<BlockProjection>& surfaces,
                             const Projection& project_c, int nb, double* v,
                             double* e, vector<bool>& done, int iters, double tol,
                             long* steps) {
  if(surfaces.empty())
    return douglas_rachford(s,project_c,nb,v,e,done,iters,tol,steps);
  int n = s.n, K = 1+(int)surfaces.size();
  long work = 0;
  size_t block = (size_t)n*nb;
  BlockProjection sphere = sphere_projection(s);
  vector<double> p((size_t)K*block);  // e_j = P_j(v_j), the sphere's first
  vector<double> y(n);                // a mean, then P_C of it
  done.assign(nb,false);
  for(int it=0;it<iters;++it) {
    memcpy(p.data(),v,sizeof(double)*K*block);
    sphere(p.data(),n,nb);
    for(int k=1;k<K;++k) surfaces[k-1](&p[k*block],n,nb);
    int live = 0;
    for(int t=0;t<nb;++t) {
      if(done[t]) continue;
      ++work;
      size_t col = (size_t)t*n;
      double all = 0.0, off = 0.0;
      for(int i=0;i<n;++i) {
        double sum = 0.0;
        for(int k=0;k<K;++k) sum += p[k*block+col+i];
        y[i] = sum/K;
      }
      project_c(y.data(),n);
      for(int k=0;k<K;++k)
        for(int i=0;i<n;++i) {
          double ek = p[k*block+col+i];
          all += ek*ek;
          off += (y[i]-ek)*(y[i]-ek);
        }
      if(off<=tol*tol*all) { done[t] = true; continue; }
      for(int i=0;i<n;++i) {
        double sum = 0.0;
        for(int k=0;k<K;++k) sum += 2.0*p[k*block+col+i]-v[k*block+col+i];
        y[i] = sum/K;
      }
      project_c(y.data(),n);
      for(int k=0;k<K;++k)
        for(int i=0;i<n;++i) v[k*block+col+i] += y[i]-p[k*block+col+i];
      ++live;
    }
    if(!live) break;
  }
  if(steps) *steps += work;
  memcpy(e,p.data(),sizeof(double)*block);
  int n_done = 0;
  for(int t=0;t<nb;++t) if(done[t]) ++n_done;
  return n_done;
}
