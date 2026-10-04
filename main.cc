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
#include "wrapper.h"

using namespace std;


// print_clique() prints a provided clique and its total weight
// in a file along with the graph header
void print_clique (
  const char* filename, const char* header,
  list<int>& clique, double clique_weight, unsigned char from1
) {
  FILE* file=fopen(filename,"w");
  fputs(header,file);
  fprintf(file,"s %lg\n",clique_weight);
  for(list<int>::iterator i=clique.begin();i!=clique.end();++i)
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

      // experimental: anchor the wrapper on the incumbent clique (see
      // anchor_wrapper()).  Each further pass re-anchors on what the previous
      // one found, and a pass that finds nothing better ends the loop, since
      // the same anchor would only repeat it.  QMS_ANCHOR_WARM keeps the first
      // pass on the standard wrapper and modifies the wrapper from the second
      // on, so that the first anchor is the clique the unmodified method ends
      // with rather than the one the greedy stage found.  It defers
      // QMS_PERTURB and QMS_ICE in the same way, so with QMS_PERTURB alone it
      // runs the matching control: a second pass on a random wrapper.
      const char* anchor = getenv("QMS_ANCHOR");
      const char* perturb = getenv("QMS_PERTURB");
      const char* ice = getenv("QMS_ICE");
      bool warm = (anchor!=NULL || perturb!=NULL || ice!=NULL) &&
        getenv("QMS_ANCHOR_WARM")!=NULL;
      int anchored_passes = anchor==NULL ? 1 :
        (getenv("QMS_ANCHOR_PASSES")?atoi(getenv("QMS_ANCHOR_PASSES")):1);
      if(anchored_passes<1) anchored_passes = 1;
      int n_passes = anchored_passes+(warm?1:0);

      for(int pass=0;pass<n_passes;++pass) {
        build_wrapper(g,info,a);
        bool modify = pass>0 || !warm;

        // experimental: perturb the free entries of the clique wrapper
        if(perturb!=NULL && modify) {
          const char* mode = getenv("QMS_PMODE");
          perturb_wrapper(g,info,a,atof(perturb),
            mode!=NULL && strcmp(mode,"unif")==0,
            getenv("QMS_SEED")?strtoull(getenv("QMS_SEED"),NULL,10):1ULL);
        }

        bool anchored = false;
        if(anchor!=NULL && modify) {
          anchored = anchor_wrapper(g,info,a,atof(anchor));
          if(!anchored && pass>0) break;
        }
        // experimental: one step of lambda_max minimization, after the
        // anchoring when there is one, on the plain wrapper otherwise
        if(ice!=NULL && modify && (anchored || anchor==NULL))
          ice_step(g,info,a,anchor!=NULL ? atof(anchor) : 1.0,ice,anchored);
        double start = info.lower_clique_bound;
        qualex_ms(info,a);
        if(getenv("QMS_STATS")!=NULL)
          fprintf(stderr,"PASS %d anchored=%d %g -> %g\n",
                  pass, (int)anchored, start, info.lower_clique_bound);
        if(anchored && !(info.lower_clique_bound>start)) break;
      }

      delete[] a;

      if(info.lower_clique_bound>clique_weight) {
        clique_weight = info.lower_clique_bound;
        clique.erase(clique.begin(),clique.end());
        for(list<int>::iterator i=info.clique.begin();i!=info.clique.end();++i)
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
    print_clique(sol_filename,g.header.c_str(),clique,clique_weight,from1);
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
