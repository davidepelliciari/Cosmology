void cbl::ImportanceSampler(const std::string file_name_A, const std::string file_name_B, const std::vector<int> usecols_A, const std::vector<int> usecols_B, const int skip_nlines_A, const int skip_nlines_B, const bool is_chi2_A, const bool is_chi2_B, const size_t distNum, const double sigmaw_cut){

  std::vector<std::vector<double>> chain_A = cbl::read_file(file_name_A, "", usecols_A, skip_nlines_A);
  std::vector<std::vector<double>> chain_B = cbl::read_file(file_name_B, "", usecols_B, skip_nlines_B);

  // numbers of parameters used
  if(usecols_A.size() != usecols_B.size()) exit(0);

  const size_t N_Params = usecols_A.size()-1;

  coutCBL  << "Number of parameters: " << N_Params << endl;

  size_t length_chain_A = chain_A[0].size();
  size_t length_chain_B = chain_B[0].size();

  size_t length_comb = length_chain_A+length_chain_B;

  coutCBL  << "Length of chain A: " << length_chain_A << endl;
  coutCBL  << "Length of chain B: " << length_chain_B << endl;
  coutCBL  << "Length of combined chain: " << length_comb << endl << endl;

  if(is_chi2_A==true){
    for(size_t ii=0; ii<length_chain_A; ii++){
      chain_A[N_Params][ii] = -0.5*chain_A[N_Params][ii];
    }
  }

  if(is_chi2_B==true){
    for(size_t ii=0; ii<length_chain_B; ii++){
      chain_B[N_Params][ii] = -0.5*chain_B[N_Params][ii];
    }
  }

  double min_logpostA = *min_element(chain_A[N_Params].begin(), chain_A[N_Params].end());
  double max_logpostA = *max_element(chain_A[N_Params].begin(), chain_A[N_Params].end());
  double min_logpostB = *min_element(chain_B[N_Params].begin(), chain_B[N_Params].end());
  double max_logpostB = *max_element(chain_B[N_Params].begin(), chain_B[N_Params].end());

  coutCBL << "Min. logPosterior_A: " << min_logpostA << endl;
  coutCBL << "Max. logPosterior_A " << max_logpostA << endl;
  coutCBL << "Min. logPosterior_B: " << min_logpostB << endl;
  coutCBL << "Max. logPosterior_B " << max_logpostB << endl << endl;

  double delta = 0;

  std::vector<std::vector<double>> chain_AB (N_Params+1, std::vector<double> (length_comb, 0.));
  std::vector<std::vector<double>> chain_BA (N_Params+1, std::vector<double> (length_comb, 0.));

  std::vector<std::vector<double>> extremals(2, std::vector<double> (N_Params, 0));

    for(size_t ii=0; ii<length_chain_A; ii++){
        for(size_t N=0; N<N_Params+1; N++){
          chain_AB[N][ii] = chain_A[N][ii];
          chain_BA[N][ii+length_chain_B] = chain_A[N][ii];
        }
      }

    // PUT THE 1ST CHAIN SECOND

    for(size_t ii=0; ii<length_chain_B; ii++){
        for(size_t N=0; N<N_Params+1; N++){
          chain_AB[N][ii+length_chain_A] = chain_B[N][ii];
          chain_BA[N][ii] = chain_B[N][ii];
        }
      }

    //FIND THE EXTREMALS FOR THE PARAMETERS IN THE CHAIN

    for(size_t j=0; j<N_Params; j++){

      extremals[0][j] = *max_element(chain_AB[j].begin(),chain_AB[j].end());
      extremals[1][j] = *min_element(chain_AB[j].begin(), chain_AB[j].end());

    }

    //RESCALING CHAINS TO -100 --> 100

    for(size_t N=0; N<N_Params; N++){
      delta = extremals[0][N]-extremals[1][N];
      for(size_t ii = 0; ii<length_comb; ii++){
        chain_AB[N][ii] = 100. - (200./(delta))*(extremals[0][N] - chain_AB[N][ii]);
        chain_BA[N][ii] = 100. - (200./(delta))*(extremals[0][N] - chain_BA[N][ii]);
      }
    }

  // set the parameters for the chain mesh
  std::vector<double> center(N_Params);
  std::vector<long> close;
  std::vector<double> distances;
  std::vector<double> indices;
  double dist , denom, sum_posterior = 0.;
  size_t cnt=0;      // forse lo tolgo

  double rMAX = 3.;
  long dim_grid = 150.;
  double cell_size = 200./dim_grid;
  long nMAX = dim_grid+10;

  std::vector<double> logPA_pointsB(length_chain_B, min_logpostA);
  std::vector<double> logPB_pointsA(length_chain_A, min_logpostB);

  // DO the chain mesh for A+B

  cbl::chainmesh::ChainMesh ch_mesh_AB(cell_size,N_Params);
  ch_mesh_AB.create_chain_mesh(chain_AB, rMAX, 0, nMAX);

  for(size_t ii=0; ii<length_chain_A;ii++){
    sum_posterior = 0.;
    denom = 0.;
    cnt=0;

    // compute the ii-th center
    for(size_t N=0; N<N_Params; N++){
      center[N] = chain_AB[N][ii];
    }

    close = ch_mesh_AB.close_objects(center, length_chain_A);
    if(close.size()>0){
      for (auto&& k : close) {
        dist = 0.;

        // euclidean distance
        for(size_t N=0; N<N_Params; N++){
          dist += pow((center[N]-chain_AB[N][k]), 2);
        }

        dist=sqrt(dist);

        distances.push_back(dist);
        indices.push_back(k);
      }

      cbl::sort_2vectors(distances.begin(),indices.begin(), distances.size());
      if(indices.size() > distNum){
        for(auto&&k : indices){
          sum_posterior += chain_AB[N_Params][k];
          denom++;
          cnt++;
          if(cnt>distNum) break;
       }
     }
     else
     {
       for(auto&&k : indices){
         sum_posterior += chain_AB[N_Params][k];
         denom++;
         cnt++;
         if(cnt>indices.size()) break;
       }
     }

      if(denom!=0){
       logPB_pointsA[ii] = sum_posterior/denom;
      }
    }

    distances.clear();
    indices.clear();
  }

  // DO the chain mesh for A+B

  cbl::chainmesh::ChainMesh ch_mesh_BA(cell_size,N_Params);
  ch_mesh_BA.create_chain_mesh(chain_BA, rMAX, 0, nMAX);

  for(size_t ii=0; ii<length_chain_B;ii++){
    sum_posterior = 0.;
    denom = 0.;
    cnt=0;

    // compute the ii-th center
    for(size_t N=0; N<N_Params; N++){
      center[N] = chain_BA[N][ii];
    }

    close = ch_mesh_BA.close_objects(center, length_chain_B);
    if(close.size()>0){
      for (auto&& k : close) {
        dist = 0.;

        // euclidean distance
        for(size_t N=0; N<N_Params; N++){
          dist += pow((center[N]-chain_BA[N][k]), 2);
        }

        dist=sqrt(dist);

        distances.push_back(dist);
        indices.push_back(k);
      }

      cbl::sort_2vectors(distances.begin(),indices.begin(), distances.size());
      if(indices.size() > distNum){
        for(auto&&k : indices){
          sum_posterior += chain_BA[N_Params][k];
          denom++;
          cnt++;
          if(cnt>distNum) break;
       }
     }
     else
     {
       for(auto&&k : indices){
         sum_posterior += chain_BA[N_Params][k];
         denom++;
         cnt++;
         if(cnt>indices.size()) break;
       }
     }

     if(denom!=0){
      logPA_pointsB[ii] = sum_posterior/denom;
     }
   }

   distances.clear();
   indices.clear();
 }

  // compute importance weights

  double max_logPB = *max_element(logPB_pointsA.begin(), logPB_pointsA.end());
  double max_logPA = *max_element(logPA_pointsB.begin(), logPA_pointsB.end());

  double shift_A = max_logpostA - max_logPB;
  double shift_B = max_logpostB - max_logPA;

  coutCBL << "shift_A: " << shift_A << endl;
  coutCBL << "shift_B: " << shift_B << endl << endl;

  std::vector<double> weights_A(length_chain_A);
  std::vector<double> weights_B(length_chain_B);

  for(size_t i=0; i<length_chain_A; i++){
    weights_A[i] = exp(logPB_pointsA[i]-chain_A[N_Params][i]+shift_A);
  }

  for(size_t i=0; i<length_chain_B; i++){
    weights_B[i] = exp(logPA_pointsB[i]-chain_B[N_Params][i]+shift_B);
  }

  coutCBL << "Min. importance weight for A+B: " << *min_element(weights_A.begin(), weights_A.end()) << endl;
  coutCBL << "Max. importance weight for A+B: " << *max_element(weights_A.begin(), weights_A.end()) << endl;
  coutCBL << "Min. importance weight for B+A: " << *min_element(weights_B.begin(), weights_B.end()) << endl;
  coutCBL << "Max. importance weight for B+A: " << *max_element(weights_B.begin(), weights_B.end()) << endl << endl;

  coutCBL << "Cutting weights distributions over " << sigmaw_cut << "-sigma" << endl;

  // cutting weights distribution over sigmaw_cut sigmas

  double mean_wA = cbl::mean(weights_A);
  double mean_wB = cbl::mean(weights_B);
  double sigma_wA = cbl::stddev(weights_A);
  double sigma_wB = cbl::stddev(weights_B);

  for(size_t ii=0; ii<weights_A.size(); ii++){
    if(weights_A[ii]>mean_wA+sigmaw_cut*sigma_wA) weights_A[ii] = mean_wA;
  }

  for(size_t ii=0; ii<weights_B.size(); ii++){
    if(weights_B[ii]>mean_wB+sigmaw_cut*sigma_wB) weights_B[ii] = mean_wB;
  }

  coutCBL << "Min. importance weight for A+B: " << *min_element(weights_A.begin(), weights_A.end()) << endl;
  coutCBL << "Max. importance weight for A+B: " << *max_element(weights_A.begin(), weights_A.end()) << endl;
  coutCBL << "Min. importance weight for B+A: " << *min_element(weights_B.begin(), weights_B.end()) << endl;
  coutCBL << "Max. importance weight for B+A: " << *max_element(weights_B.begin(), weights_B.end()) << endl << endl;

  std::string output_path = "./";
  std::string filename_AB = output_path+"imp_sampling_1+2_chain.dat";
  std::string filename_BA = output_path+"imp_sampling_2+1_chain.dat";
  std::string filename_final = output_path+"imp_sampling_final_chain.dat";

  std::ofstream out_AB(filename_AB, ios::trunc);
  std::ofstream out_BA(filename_BA, ios::trunc);
  std::ofstream out_final(filename_final, ios::trunc);

  coutCBL << "Writing chains in " << filename_AB << " opened." << endl;

  if(out_AB.is_open()){
    out_AB << "#steps_MCMC #parameters #logPost_1 #logPost_2(points_1) #importance_weights" << endl;
    for(size_t i=0; i<length_chain_A; i++){
      out_AB << i << " ";
      for(size_t N=0; N<N_Params+1; N++){
        out_AB << chain_A[N][i] << " ";
      }
      out_AB << logPB_pointsA[i] << " " << weights_A[i] << endl;
    }
  }

  out_AB.close();

  coutCBL << "Writing chains in " << filename_BA << " opened." << endl;

  if(out_BA.is_open()){
    out_BA << "#steps_MCMC #parameters #logPost_2 #logPost_1(points_2) #importance_weights" << endl;
    for(size_t i=0; i<length_chain_B; i++){
      out_BA << i << " ";
      for(size_t N=0; N<N_Params+1; N++){
        out_BA << chain_B[N][i] << " ";
      }
      out_BA << logPA_pointsB[i] << " " << weights_B[i] << endl;
    }
  }

  out_BA.close();

  // concatenates importance chains

  std::vector<std::vector<double>> imp_chain(N_Params+2);

  imp_chain[N_Params].insert(imp_chain[N_Params].begin(), logPB_pointsA.begin(), logPB_pointsA.end());
  imp_chain[N_Params].insert(imp_chain[N_Params].end(), logPA_pointsB.begin(), logPA_pointsB.begin());
  imp_chain[N_Params+1].insert(imp_chain[N_Params+1].begin(), weights_A.begin(), weights_A.end());
	imp_chain[N_Params+1].insert(imp_chain[N_Params+1].end(), weights_B.begin(), weights_B.end());

  for(size_t N=0; N<N_Params; N++){
    imp_chain[N].insert(imp_chain[N].begin(), chain_A[N].begin(), chain_A[N].end());
    imp_chain[N].insert(imp_chain[N].end(), chain_B[N].begin(), chain_B[N].end());
  }

  coutCBL << "Writing chains in " << filename_final << " opened." << endl;
  if(out_final.is_open()){
    out_final << "#steps_MCMC #parameters #logPost #importance_weights" << endl;
    for(size_t ii=0; ii<length_comb; ii++){
      out_final << ii << " ";
      for(size_t N=0; N<imp_chain.size(); N++){
        out_final << imp_chain[N][ii] << " ";
      }
      out_final << endl;
    }
  }

  out_final.close();

}
