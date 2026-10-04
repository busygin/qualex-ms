/***********************************************************************
!! dr.cc: Douglas-Rachford iterations between a sphere and a set given !!
!! by its projection (proposed by S. Busygin).                         !!
***********************************************************************/

#include <math.h>
#include <string.h>
#include <cblas.h>

#include "dr.h"

using namespace std;

void clip_negative(double* y, int n) {
  for(int i=0;i<n;++i) if(y[i]<0.0) y[i] = 0.0;
}

int douglas_rachford(const Sphere& s, const Projection& project_c,
                     int nb, double* v, double* e, vector<bool>& done,
                     int iters, double tol) {
  int n = s.n, k = s.k;
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
  int n_done = 0;
  for(int t=0;t<nb;++t) if(done[t]) ++n_done;
  return n_done;
}
