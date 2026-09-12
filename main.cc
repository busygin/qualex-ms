/*****************************************************************************
!!  This is the main() function of QUALEX-MS solver                         !!
!!  Copyright (c) Stanislav Busygin, 2000-2007. All rights reserved.        !!
!!                                                                          !!
!! This software is distributed AS IS. NO WARRANTY is expressed or implied. !!
!! The author grants a permission for everyone to use and distribute this   !!
!! software free of charge for research and educational purposes.           !!
!! Any COMMERCIAL usage of this software is PROHIBITED without a written    !!
!! permission of the copyright holder.                                      !!
!!                                                                          !!
!! Please send any inquiry to <busygin@gmail.com> and visit                 !!
!! Stas Busygin's NP-completeness page: <http://www.busygin.dp.ua/npc.html> !!
*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "graph.h"
#include "greedy_clique.h"
#include "preproc_clique.h"
#include "qualex.h"

// perturb_wrapper() replaces the zero entries of the weighted adjacency matrix
// on non-adjacent vertex pairs, which are the free parameters of the clique
// wrapper: Proposition 2 of the clique wrapper note admits any value below
// z_i z_j there.
//
// Every value this accepts is safe.  Lowering an entry strictly decreases
// x^T H x wherever x_i x_j > 0 and leaves every clique-supported point exactly
// where it was, so a value <= 0 carries the generalized Motzkin-Straus theorem
// over unchanged, optimum value and clique optima together.  A strictly
// negative one goes further: the complete multipartite optima of Theorem 4 all
// contain a non-adjacent pair and so become strictly suboptimal, leaving the
// maximum weight clique indicators as the only global optima.  A perturbed run
// therefore never reports a clique that is not one.
//
// The reason to do it is that the perturbation moves the spectrum and the
// eigenvectors, so the trust region stage visits different stationary points,
// and different wrappers expose different cliques.  Measured on the current
// solver at eta = 0.01 over three seeds: keller5 26 -> 27, which is optimal,
// on two of them, and MANN_a45 342 -> 344 against a best known 345.  Nothing
// regressed at that scale on any instance tried.
//
// Keep eta small.  At 0.5 the same two gains survive but several instances
// fall over -- san400_0.7_3 22 -> 18, gen400_p0.9_55 53 -> 51 on every seed,
// san200_0.9_3 44 -> 43 -- which is what the loss of nonnegativity in A^(w)
// costs: the paper's argument for why the stationary points stay roughly
// nonnegative leans on it.  The response is not monotone in eta either
// (keller5 on seed 1 reaches 27 at 0.01 through 0.35 and again at 0.8, but not
// at 0.5 or 1.5), so the effect is diversity among wrappers rather than a
// threshold to tune past.  Run several and keep the best; a single wrapper is
// not a better default, and this stays off unless asked for.
//
// Small perturbations only became useful once the cluster tolerances were put
// on the right scale.  Against the old absolute cut of 1e-5 on the summed
// linear form, an eta of 0.05 split the eigenvalues but left every cluster
// classified degenerate, because the coefficients it created were real but
// tiny; against the present n*eps*||hatb|| they count.
//
// The uniform mode is a control rather than a strategy.  eps_ij = -eta z_i z_j
// makes H = (1+eta) H_0 - eta z z^T, and the projection onto z^T x = 1 removes
// the rank one term, leaving the eigenvectors alone and the eigenvalues
// affinely mapped; substituting into y_i = c_i/(mu - lambda_i) reproduces the
// same stationary points.  It should match an unperturbed run exactly, and
// exists to check that it does.
void perturb_wrapper(Graph& g, MaxCliqueInfo& info, double* a, double eta,
                     bool uniform, unsigned long long seed) {
  int& n = g.n;
  for(int i=0;i<n;i++) for(int j=0;j<i;j++) {
    if(g.mates[i].at(j)) continue;
    // Knuth MMIX linear congruential generator, so that a seed reproduces a
    // wrapper exactly whatever the platform's long happens to be
    seed = seed*6364136223846793005ULL + 1442695040888963407ULL;
    double u = uniform ? 1.0 :
      (double)((seed>>11)&0xFFFFFFFFULL)/4294967296.0;
    a[i*n+j] = a[j*n+i] = -eta*u*info.sqrtw[i]*info.sqrtw[j];
  }
}

// print_clique() prints a provided clique and its total weight
// in a file along with the graph header
void print_clique (
  const char* filename, const char* header,
  list<int>& clique, double clique_weight, unsigned char from1
) {
  FILE* file=fopen(filename,"w");
  fputs(header,file);
  fprintf(file,"s %lg\n",clique_weight);
  for(list<int>::iterator i=clique.begin();i!=clique.end();i++)
    fprintf(file, "v%11d\n", *i + from1);
  fclose(file);
}

int main(int argc,char** argv) {
  puts(
    "QUick ALmost EXact maximum weight clique solver, ver. 1.2-MS\n\n"
    "Copyright (c) Stanislav Busygin, 2000-2007. All rights reserved.\n\n"
    "This software is distributed AS IS. NO WARRANTY is expressed or implied.\n"
    "The author grants a permission for everyone to use and distribute this\n"
    "software free of charge for research and educational purposes.\n"
    "Any COMMERCIAL usage of this software is PROHIBITED without a written\n"
    "permission of the copyright holder.\n\n"
    "Please send any inquiry to <busygin@gmail.com> and visit\n"
    "Stas Busygin's NP-completeness page: <http://www.busygin.dp.ua/npc.html>\n"
  );

  char* name=NULL;
  char* weights_name=NULL;
  bool for_clique=true;
  unsigned char from1 = '\0';
  char* p;

  // get running parameters
  while(argc-->=2) {
    p=*++argv;
    switch(p[0]) {
      case '-':
        switch(p[1]) {
          case 'c':
            for_clique=false;
            break;
          case '1':
            from1 = '\0';
            break;
          case 'w':
            weights_name=p+2;
        }
        break;
      case '+':
        switch(p[1]) {
          case 'c':
            for_clique=true;
            break;
          case '1':
            from1 = '\1';
        }
        break;
      default:
        name=p;
    }
  }

  if(name!=NULL) {  // parameters are valid
    printf("%s mode.\n", for_clique?"CLIQUE":"MIS");

    // load the graph from a DIMACS file
    Graph g(name,weights_name,!for_clique);

    // note the start time
    time_t time1,time2;
    time(&time1);

    // preprocess
    vector<int> residual;
    list<int> preselected, clique;
    double clique_weight;
    double preselected_weight = preproc_clique (
      g,residual,preselected,clique_weight,clique
    );

    // if the instance is not reduced completely, apply QUALEX-MS
    if(!residual.empty()) {
      MaxCliqueInfo info(g,for_clique);
      meta_greedy_clique(info);

      int& n=g.n;
      double* a = new double[n*n];
      memset(a,0,sizeof(double)*n*n);
      int i,j;

      for(i=0;i<n;i++) {
        a[i*(n+1)] = g.weights[i]-info.w_min;
        bit_iterator bi(g.mates[i]);
        while((j=bi.next())>-1) {
          if(j>i) break;
          a[i*n+j] = a[j*n+i] = info.sqrtw[i]*info.sqrtw[j];
        }
      }

      // experimental: perturb the free entries of the clique wrapper
      if(getenv("QMS_PERTURB")!=NULL) {
        const char* mode = getenv("QMS_PMODE");
        perturb_wrapper(g,info,a,atof(getenv("QMS_PERTURB")),
          mode!=NULL && strcmp(mode,"unif")==0,
          getenv("QMS_SEED")?strtoull(getenv("QMS_SEED"),NULL,10):1ULL);
      }

      qualex_ms(info,a);

      delete[] a;

      if(info.lower_clique_bound>clique_weight) {
        clique_weight = info.lower_clique_bound;
        clique.erase(clique.begin(),clique.end());
        for(list<int>::iterator i=info.clique.begin();i!=info.clique.end();i++)
          clique.push_back(residual[*i]);
      }
    }

    // join with the earlier preselected vertices
    clique.splice(clique.begin(),preselected);
    clique_weight += preselected_weight;

    // note the finish time
    time(&time2);

    // print results
    printf (
      "%s: %s_w >= %lg, time=%lg sec.\n",
      name, for_clique?"omega":"alpha", clique_weight, difftime(time2,time1)
    );

    int length=strlen(name);
    char* sol_filename=new char[length+5];
    memcpy(sol_filename, name, length+1);
    char* p=strstr(sol_filename,for_clique?".clq":".mis");
    if(!p)p=sol_filename+length;
    strcpy(p,".sol");
    print_clique(sol_filename,g.header,clique,clique_weight,from1);
    delete[] sol_filename;
  } else puts(
    "Syntax: qualex-ms [<flag>] <dimacs_binary_file> [-w<weights_file>]\n"
    "Flags:\n"
    "+c: look for maximum clique (default)\n"
    "-c: look for maximum independent set\n"
    "+1: vertex numbers in solution file go from 1\n"
    "-1: vertex numbers in solution file go from 0 (default)\n"
    "weights_file: a text file for list of vertex weights (reals)\n"
  );

  return 0;
}
