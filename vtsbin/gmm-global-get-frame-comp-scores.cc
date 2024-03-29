// gmmbin/gmm-global-get-frame-comp-likes.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/model-common.h"
#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-full-gmm.h"

#include "vts/vts-first-order.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Print out per-frame pre-Gaussian (log-)likelihoods or posteriors for each utterance, as an archive\n"
            "of matrices of floats.\n"
            "Usage:  gmm-global-get-frame-comp-scores [options] <model-in> <feature-rspecifier> "
            "<likes-out-wspecifier> [<noise-rspecifier>]\n"
            "e.g.: gmm-global-get-frame-comp-scores 1.mdl scp:train.scp ark:1.likes\n";

    ParseOptions po(usage);

    bool apply_log = true;
    po.Register("apply-log", &apply_log,
                "Output log score or not.");

    std::string type = "likelihood";
    po.Register("type", &type, "Output 'likelihood' or 'posterior'.");

    int32 num_cepstral = 13;
    int32 num_fbank = 26;
    BaseFloat ceplifter = 22;

    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");

    po.Read(argc, argv);

    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    bool have_noise = false;
    if (po.NumArgs() == 4) {
      have_noise = true;
    }

    bool out_likes = true;
    if (type != "likelihood" && type != "posterior"){
      KALDI_ERR<< "Invalid output feature type: " << type << "!";
    } else if (type == "posterior"){
      out_likes = false;
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        likes_wspecifier = po.GetArg(3),
        noise_rspecifier = po.GetOptArg(4);

    DiagGmm gmm;
    {
      bool binary_read;
      Input ki(model_filename, &binary_read);
      gmm.Read(ki.Stream(), binary_read);
    }

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    double tot_like = 0.0, tot_frames = 0.0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_rspecifier);
    BaseFloatMatrixWriter likes_writer(likes_wspecifier);
    int32 num_done = 0, num_err = 0;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      int32 file_frames = mat.NumRows();
      Matrix<BaseFloat> scores(file_frames, gmm.NumGauss());

      if (!have_noise) {
        // no noise..
        for (int32 i = 0; i < file_frames; i++) {
          Vector<BaseFloat> loglikes;
          if(out_likes) {
            gmm.LogLikelihoods(mat.Row(i), &loglikes);
          }else {
            gmm.ComponentPosteriors(mat.Row(i), &loglikes);
          }
          if (!apply_log) {
            loglikes.ApplyExp();
          }
          scores.CopyRowFromVec(loglikes, i);
        }
      } else {
        // have noise

        if (!noiseparams_reader.HasKey(key + "_mu_h")
            || !noiseparams_reader.HasKey(key + "_mu_z")
            || !noiseparams_reader.HasKey(key + "_var_z")) {
          KALDI_ERR<< "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
        }

        int feat_dim = mat.NumCols();
        if (feat_dim != 39) {
          KALDI_ERR << "Could not decode the features, only 39D MFCC_0_D_A is supported!";
        }

        /************************************************
         Extract the noise parameters
         *************************************************/

        Vector<double> mu_h(noiseparams_reader.Value(key + "_mu_h"));
        Vector<double> mu_z(noiseparams_reader.Value(key + "_mu_z"));
        Vector<double> var_z(noiseparams_reader.Value(key + "_var_z"));

        DiagGmm noise_gmm;
        noise_gmm.CopyFromDiagGmm(gmm);

        std::vector<Matrix<double> > Jx(gmm.NumGauss()), Jz(gmm.NumGauss());  // not necessary for compensation only
        CompensateDiagGmm(mu_h, mu_z, var_z, num_cepstral, num_fbank, dct_mat, inv_dct_mat, noise_gmm, Jx, Jz);

        for (int32 i = 0; i < file_frames; i++) {
          Vector<BaseFloat> loglikes;
          if(out_likes){
            noise_gmm.LogLikelihoods(mat.Row(i), &loglikes);
          }else{
            noise_gmm.ComponentPosteriors(mat.Row(i), &loglikes);
          }
          if (!apply_log) {
            loglikes.ApplyExp();
          }
          scores.CopyRowFromVec(loglikes, i);
        }
      }

      tot_like += scores.Sum();
      tot_frames += file_frames;
      likes_writer.Write(key, scores);
      num_done++;
    }
    KALDI_LOG<< "Done " << num_done << " files; " << num_err
    << " with errors.";
    KALDI_LOG<< "Overall likelihood per "
    << "frame = " << (tot_like/tot_frames) << " over " << tot_frames
    << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
