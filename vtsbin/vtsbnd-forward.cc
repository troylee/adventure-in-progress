/*
 * dbnvts-forward.cc
 *
 *  Created on: Oct 29, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Estimate the new decision boundaries per utterance and forward
 *  through the whole network.
 *
 *  The boundary is estimated from the two unconstrained Gaussians
 *  using 1st order VTS approximation.
 *
 */

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"

#include "vts/vts-first-order.h"
#include "vts/dbnvts-first-order.h"
#include "vts/vtsbnd-first-order.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Estimate the new decision boundaries and perform forward pass through Neural Network.\n"
            "Usage:  vtsbnd-forward [options] <biasedlinearity-layer-in> <pos-model-in> <neg-model-in>"
            " <back-nnet-in> <pos2neg-prior-in> <var-scale-in> <feature-rspecifier>"
            " <noise-model-rspecifier> <feature-wspecifier>\n"
            "e.g.: \n"
            " vtsbnd-forward bl_layer pos.mdl neg.mdl nnet.back ark:features.ark ark:noise_params.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    bool compensate_var = true;
    po.Register(
        "compensate-var",
        &compensate_var,
        "Whether apply VTS compensation to covariance. Mean will always be compensated");

    bool clean_cmvn = true;
    po.Register(
        "clean-cmvn",
        &clean_cmvn,
        "Inputs are clean CMVN normalized features or noise compensated CMVN normalized features");

    BaseFloat interpolate_weight = 1.0;  // use new estimation
    po.Register("interpolate-weight", &interpolate_weight,
                "Interpolation weight for the new boundary estimation");

    std::string cmvn_stats_rspecifier;
    po.Register("cmvn-stats", &cmvn_stats_rspecifier,
                "rspecifier for global CMVN feature normalization");

    int32 num_frames = 9;
    po.Register("num-frames", &num_frames,
                "Number of frames for the input feature");

    int32 num_cepstral = 13;
    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");

    int32 num_fbank = 26;
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");

    BaseFloat ceplifter = 22;
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");

    bool use_var_scale = false;
    po.Register("use-var-scale", &use_var_scale,
                "Apply the variance scale to the new weight estimation");

    std::string class_frame_counts;
    po.Register("class-frame-counts", &class_frame_counts,
                "Counts of frames for posterior division by class-priors");

    BaseFloat prior_scale = 1.0;
    po.Register(
        "prior-scale",
        &prior_scale,
        "scaling factor of prior log-probabilites given by --class-frame-counts");

    bool apply_log = false, silent = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    bool no_softmax = false;
    po.Register(
        "no-softmax",
        &no_softmax,
        "No softmax on MLP output. The MLP outputs directly log-likelihoods, log-priors will be subtracted");

    po.Register("silent", &silent, "Don't print any messages");

    po.Read(argc, argv);

    if (po.NumArgs() != 9) {
      po.PrintUsage();
      exit(1);
    }

    std::string bl_layer_filename = po.GetArg(1), pos_model_filename =
        po.GetArg(2), neg_model_filename = po.GetArg(3), nnet_filename = po
        .GetArg(4), pos2neg_prior_filename = po.GetArg(5), var_scale_filename =
        po.GetArg(6), feature_rspecifier = po.GetArg(7),
        noise_params_rspecifier = po.GetArg(8), feature_wspecifier = po.GetArg(
            9);

    // biased linearity layer weight and bias
    Matrix<BaseFloat> linearity;
    Vector<BaseFloat> bias;
    {
      bool binary;
      Input ki(bl_layer_filename, &binary);
      if (!ReadBiasedLinearityLayer(ki.Stream(), binary, linearity, bias)) {
        KALDI_ERR << "Load biased linearity layer from " << bl_layer_filename
            << " failed!";
      }
    }

    // positive AM Gmm
    AmDiagGmm pos_am_gmm;
    {
      bool binary;
      Input ki(pos_model_filename, &binary);
      pos_am_gmm.Read(ki.Stream(), binary);
    }

    // negative AM Gmm
    AmDiagGmm neg_am_gmm;
    {
      bool binary;
      Input ki(neg_model_filename, &binary);
      neg_am_gmm.Read(ki.Stream(), binary);
    }

    // converting the normalized gmms to original feature space
    // if necessary
    int32 frame_dim = linearity.NumCols() / num_frames;
    KALDI_ASSERT(frame_dim * num_frames == linearity.NumCols());
    // global_* are expanded mean and standard deviation for feature normalization
    // frame_* are unexpanded, single frame mean and covaraince (i.e. std*std)
    Vector<double> global_mean(linearity.NumCols()), frame_mean(frame_dim);
    Vector<double> global_std(linearity.NumCols()), frame_var(frame_dim),
        frame_std(frame_dim);
    if (cmvn_stats_rspecifier != "") {
      // convert the models back to the original feature space
      RandomAccessDoubleMatrixReader cmvn_reader(cmvn_stats_rspecifier);
      if (!cmvn_reader.HasKey("global")) {
        KALDI_ERR << "No normalization statistics available for key global";
      }
      const Matrix<double> &stats = cmvn_reader.Value("global");
      // convert stats to mean and std
      int32 dim = stats.NumCols() - 1;
      double count = stats(0, dim);
      if (count < 1.0)
        KALDI_ERR
            << "Insufficient stats for cepstral mean and variance normalization: "
            << "count = " << count;

      KALDI_ASSERT(global_mean.Dim() % dim == 0);

      for (int32 i = 0; i < dim; ++i) {
        frame_mean(i) = stats(0, i) / count;
        frame_var(i) = (stats(1, i) / count) - frame_mean(i) * frame_mean(i);
        frame_std(i) = sqrt(frame_var(i));
      }
      for (int32 i = 0; i < linearity.NumCols(); ++i) {
        global_mean(i) = frame_mean(i % dim);
        global_std(i) = frame_std(i % dim);
      }

      NormalizedGmmToGmm(global_mean, global_std, pos_am_gmm);
      NormalizedGmmToGmm(global_mean, global_std, neg_am_gmm);
    }

    // noisy models
    AmDiagGmm pos_noise_am, neg_noise_am;

    Nnet nnet_back;
    nnet_back.Read(nnet_filename);

    // positive to negative prior ratio
    Vector<BaseFloat> pos2neg_log_prior_ratio(linearity.NumRows(), kSetZero);
    Matrix<double> prior_stats;
    {
      bool binary;
      Input ki(pos2neg_prior_filename, &binary);
      prior_stats.Read(ki.Stream(), binary);
      KALDI_ASSERT(
          prior_stats.NumRows()==2 && prior_stats.NumCols()==linearity.NumRows());
    }
    for (int32 i = 0; i < linearity.NumRows(); ++i) {
      pos2neg_log_prior_ratio(i) = log(prior_stats(0, i) / prior_stats(1, i));
    }

    // variance scale factors
    Vector<double> var_scale;
    {
      bool binary;
      Input ki(var_scale_filename, &binary);
      var_scale.Read(ki.Stream(), binary);
      KALDI_ASSERT(var_scale.Dim() == linearity.NumRows());
    }
    if (!use_var_scale) {  // if no scaling is applied, set them to 1
      for (int32 i = 0; i < linearity.NumRows(); ++i) {
        var_scale(i) = 1.0;
      }
    }

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_params_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feat_dev, nnet_out_dev;
    Matrix<BaseFloat> nnet_out_host;

    Matrix<BaseFloat> vts_linearity(linearity);
    Vector<BaseFloat> vts_bias(bias);

    // Read the class-counts, compute priors
    Vector<BaseFloat> tmp_priors;
    CuVector<BaseFloat> priors;
    if (class_frame_counts != "") {
      Input in;
      in.OpenTextMode(class_frame_counts);
      tmp_priors.Read(in.Stream(), false);
      in.Close();

      BaseFloat sum = tmp_priors.Sum();
      tmp_priors.Scale(1.0 / sum);
      if (apply_log || no_softmax) {
        tmp_priors.ApplyLog();
        tmp_priors.Scale(-prior_scale);
      } else {
        tmp_priors.ApplyPow(-prior_scale);
      }

      // push priors to GPU
      priors.CopyFromVec(tmp_priors);
    }

    Timer tim;
    if (!silent)
      KALDI_LOG << "DBNVTS FEEDFORWARD STARTED";

    int32 num_done = 0;
    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read the features
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &feat = feature_reader.Value();

      // read noise parameters
      if (!noiseparams_reader.HasKey(key + "_mu_h")
          || !noiseparams_reader.HasKey(key + "_mu_z")
          || !noiseparams_reader.HasKey(key + "_var_z")) {
        KALDI_ERR
            << "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
      }
      Vector<double> mu_h(noiseparams_reader.Value(key + "_mu_h"));
      Vector<double> mu_z(noiseparams_reader.Value(key + "_mu_z"));
      Vector<double> var_z(noiseparams_reader.Value(key + "_var_z"));
      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG << "Additive Noise Mean: " << mu_z;
        KALDI_LOG << "Additive Noise Covariance: " << var_z;
        KALDI_LOG << "Convoluational Noise Mean: " << mu_h;
      }

      // compensate the postive and negative gmm models
      pos_noise_am.CopyFromAmDiagGmm(pos_am_gmm);
      CompensateMultiFrameGmm(mu_h, mu_z, var_z, compensate_var, num_cepstral,
                              num_fbank, dct_mat, inv_dct_mat, num_frames,
                              pos_noise_am);

      neg_noise_am.CopyFromAmDiagGmm(neg_am_gmm);
      CompensateMultiFrameGmm(mu_h, mu_z, var_z, compensate_var, num_cepstral,
                              num_fbank, dct_mat, inv_dct_mat, num_frames,
                              neg_noise_am);

      // convert back to normalized feature space under noisy conditions if necessary
      if (cmvn_stats_rspecifier != "") {
        // compensate the global clean mean and variance to get the global mean and
        // variance under noisey condition

        if (clean_cmvn) {
          GmmToNormalizedGmm(global_mean, global_std, pos_noise_am);
          GmmToNormalizedGmm(global_mean, global_std, neg_noise_am);
        } else {
          Vector<double> noise_frame_mean(frame_mean);
          Vector<double> noise_frame_var(frame_var);
          Matrix<double> Jx, Jz;

          CompensateDiagGaussian(mu_h, mu_z, var_z, num_cepstral, num_fbank,
                                 dct_mat, inv_dct_mat, noise_frame_mean,
                                 noise_frame_var, Jx, Jz);

          Vector<double> noise_mean(global_mean.Dim());
          Vector<double> noise_std(global_mean.Dim());
          for (int32 i = 0; i < noise_mean.Dim(); ++i) {
            noise_mean(i) = noise_frame_mean(i % frame_dim);
            noise_std(i) = sqrt(noise_frame_var(i % frame_dim));  // convert covariance to standard deviation
          }

          GmmToNormalizedGmm(noise_mean, noise_std, pos_noise_am);
          GmmToNormalizedGmm(noise_mean, noise_std, neg_noise_am);
        }
      }

      vts_linearity.CopyFromMat(linearity);
      vts_bias.CopyFromVec(bias);
      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG << "Old weights: " << vts_linearity;
        KALDI_LOG << "Old bias:" << vts_bias;
      }

      // estimate the new decision boundary
      ComputeFirstOrderVTSGaussianBoundary(pos_am_gmm, neg_am_gmm,
                                           pos2neg_log_prior_ratio, var_scale,
                                           vts_linearity, vts_bias);

      // interpolate with the original one
      vts_linearity.Scale(interpolate_weight);
      vts_linearity.AddMat(1 - interpolate_weight, linearity);
      vts_bias.Scale(interpolate_weight);
      vts_bias.AddVec(1 - interpolate_weight, bias);

      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG << "New weights: " << vts_linearity;
        KALDI_LOG << "New bias: " << vts_bias;
      }

      // forward through the new biased linearity layer
      Matrix<BaseFloat> mat(feat.NumRows(), vts_linearity.NumRows());
      mat.AddMatMat(1.0, feat, kNoTrans, vts_linearity, kTrans, 0.0);  // w_T * feat
      mat.AddVecToRows(1.0, vts_bias);

      //check for NaN/inf
      for (int32 r = 0; r < mat.NumRows(); r++) {
        for (int32 c = 0; c < mat.NumCols(); c++) {
          BaseFloat val = mat(r, c);
          if (val != val)
            KALDI_ERR << "NaN in features of : " << key;
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR << "inf in features of : " << key;
        }
      }
      // push it to gpu
      feat_dev.CopyFromMat(mat);
      // fwd-pass
      nnet_back.Feedforward(feat_dev, &nnet_out_dev);

      // convert posteriors to log-posteriors
      if (apply_log) {
        nnet_out_dev.ApplyLog();
      }

      // divide posteriors by priors to get quasi-likelihoods
      if (class_frame_counts != "") {
        if (apply_log || no_softmax) {
          nnet_out_dev.AddVecToRows(1.0, priors, 1.0);
        } else {
          nnet_out_dev.MulColsVec(priors);
        }
      }

      //download from GPU
      nnet_out_dev.CopyToMat(&nnet_out_host);
      //check for NaN/inf
      for (int32 r = 0; r < nnet_out_host.NumRows(); r++) {
        for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
          BaseFloat val = nnet_out_host(r, c);
          if (val != val)
            KALDI_ERR << "NaN in NNet output of : " << key;
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR << "inf in NNet coutput of : " << key;
        }
      }
      // write
      feature_writer.Write(key, nnet_out_host);

      // progress log
      if (num_done % 1000 == 0) {
        if (!silent)
          KALDI_LOG << num_done << ", " << std::flush;
      }
      num_done++;
      tot_t += mat.NumRows();
    }

    // final message
    if (!silent)
      KALDI_LOG << "MLP FEEDFORWARD FINISHED " << tim.Elapsed() << "s, fps"
          << tot_t / tim.Elapsed();
    if (!silent)
      KALDI_LOG << "Done " << num_done << " files";

#if HAVE_CUDA==1
    if (!silent) CuDevice::Instantiate().PrintProfile();
#endif

    return ((num_done > 0) ? 0 : 1);
  } catch (const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}

