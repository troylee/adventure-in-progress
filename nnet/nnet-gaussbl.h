// nnet/nnet-gaussbl.h

#ifndef KALDI_NNET_GAUSSBL_H
#define KALDI_NNET_GAUSSBL_H

#include "nnet/nnet-component.h"
#include "gmm/am-diag-gmm.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {

class GaussBL : public UpdatableComponent {

 public:
  GaussBL(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : UpdatableComponent(dim_in, dim_out, nnet),
        cpu_linearity_(dim_out, dim_in),
        cpu_bias_(dim_out),
        linearity_(dim_out, dim_in),
        bias_(dim_out),
        cpu_linearity_corr_(dim_out, dim_in),
        cpu_bias_corr_(dim_out),
        linearity_corr_(dim_out, dim_in),
        bias_corr_(dim_out),
        log_prior_ratio_(dim_out),
        precision_coeff_(dim_out, dim_in),
        num_cepstral_(-1),
        num_fbank_(-1),
        ceplifter_(-1),
        compensate_var_(true)
  {
  }
  ~GaussBL()
  {
  }

  ComponentType GetType() const {
    return kGaussBL;
  }

  /* Do the necessary computations to gaurantee the Gaussian representation is
   * equivalent to the original NN layer.
   */
  void CreateModel(int32 num_frame, int32 delta_order, int32 num_cepstral,
                   int32 num_fbank,
                   BaseFloat ceplifter,
                   const AmDiagGmm &pos_am,
                   const AmDiagGmm &neg_am,
                   const Matrix<BaseFloat> &weight, /* target weight matrix */
                   const Vector<BaseFloat> &bias /* target bias */) {
    num_frame_ = num_frame;
    delta_order_ = delta_order;
    num_cepstral_ = num_cepstral;
    num_fbank_ = num_fbank;
    ceplifter_ = ceplifter;
    pos_am_gmm_.CopyFromAmDiagGmm(pos_am);
    neg_am_gmm_.CopyFromAmDiagGmm(neg_am);

    KALDI_ASSERT(
        pos_am_gmm_.NumPdfs() == output_dim_ && pos_am_gmm_.Dim() == input_dim_);
    KALDI_ASSERT(
        neg_am_gmm_.NumPdfs() == output_dim_ && neg_am_gmm_.Dim() == input_dim_);

    //TODO:: derive the log prior and the precision matrix coefficients
    ComputeLogPriorAndPrecCoeff(weight, bias);

    PrepareDCTXforms();
    ConvertToNNLayer(pos_am_gmm_, neg_am_gmm_);  // generate the NN layer from the clean Gaussians
    linearity_.CopyFromMat(cpu_linearity_);
    bias_.CopyFromVec(cpu_bias_);
  }

  void ReadData(std::istream &is, bool binary) {
    ReadBasicType(is, binary, &num_frame_);
    ReadBasicType(is, binary, &delta_order_);
    ReadBasicType(is, binary, &num_cepstral_);
    ReadBasicType(is, binary, &num_fbank_);
    ReadBasicType(is, binary, &ceplifter_);

    log_prior_ratio_.Read(is, binary);
    precision_coeff_.Read(is, binary);
    pos_am_gmm_.Read(is, binary);
    neg_am_gmm_.Read(is, binary);

    KALDI_ASSERT(log_prior_ratio_.Dim() == output_dim_);
    KALDI_ASSERT(precision_coeff_.NumRows() == output_dim_ && precision_coeff_.NumCols() == input_dim_);
    KALDI_ASSERT(
        pos_am_gmm_.NumPdfs() == output_dim_ && pos_am_gmm_.Dim() == input_dim_);
    KALDI_ASSERT(
        neg_am_gmm_.NumPdfs() == output_dim_ && neg_am_gmm_.Dim() == input_dim_);

    PrepareDCTXforms();
    ConvertToNNLayer(pos_am_gmm_, neg_am_gmm_);
    linearity_.CopyFromMat(cpu_linearity_);
    bias_.CopyFromVec(cpu_bias_);

  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteBasicType(os, binary, num_frame_);
    WriteBasicType(os, binary, delta_order_);
    WriteBasicType(os, binary, num_cepstral_);
    WriteBasicType(os, binary, num_fbank_);
    WriteBasicType(os, binary, ceplifter_);
    log_prior_ratio_.Write(os, binary);
    precision_coeff_.Write(os, binary);
    pos_am_gmm_.Write(os, binary);
    neg_am_gmm_.Write(os, binary);
  }

  // CPU based forward
  void Forward(const Matrix<BaseFloat> &in, Matrix<BaseFloat> *out) {
    // precopy bias
    out->SetZero();
    out->AddVecToRows(1.0, cpu_bias_);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, cpu_linearity_, kTrans, 1.0);
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // precopy bias
    out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err,
                        CuMatrix<BaseFloat> *out_err) {
    // multiply error by weights
    out_err->AddMatMat(1.0, in_err, kNoTrans, linearity_, kNoTrans, 0.0);
  }

  void Update(const CuMatrix<BaseFloat> &input,
              const CuMatrix<BaseFloat> &err) {
    // compute gradient
    if (average_grad_) {
      linearity_corr_.AddMatMat(1.0 / input.NumRows(), err, kTrans, input,
                                kNoTrans,
                                momentum_);
      bias_corr_.AddRowSumMat(1.0 / input.NumRows(), err, momentum_);
    } else {
      linearity_corr_.AddMatMat(1.0, err, kTrans, input, kNoTrans, momentum_);
      bias_corr_.AddRowSumMat(1.0, err, momentum_);
    }

    linearity_corr_.CopyToMat(&cpu_linearity_corr_);
    bias_corr_.CopyToVec(&cpu_bias_corr_);

    // update the log ratio, the gradient equals to bias gradient
    log_prior_ratio_.AddVec(-learn_rate_, cpu_bias_corr_);

    // update var scale
    //UpdateVarScale();

    /*
     // l2 regularization
     if (l2_penalty_ != 0.0) {
     BaseFloat l2 = learn_rate_ * l2_penalty_ * input.NumRows();
     linearity_.AddMat(-l2, linearity_);
     }
     // l1 regularization
     if (l1_penalty_ != 0.0) {
     BaseFloat l1 = learn_rate_ * input.NumRows() * l1_penalty_;
     cu::RegularizeL1(&linearity_, &linearity_corr_, l1, learn_rate_);
     }
     // update
     linearity_.AddMat(-learn_rate_, linearity_corr_);
     bias_.AddVec(-learn_rate_, bias_corr_);
     */
  }

  void SetUpdateFlag(std::string flag) {

  }

  void PrepareDCTXforms();

  void SetNoise(bool compensate_var, const Vector<double> &mu_h,
                const Vector<double> &mu_z,
                const Vector<double> &var_z);

  void GetNoise(Vector<double> &mu_h, Vector<double> &mu_z,
                Vector<double> &var_z) {
    mu_h.CopyFromVec(mu_h_);
    mu_z.CopyFromVec(mu_z_);
    var_z.CopyFromVec(var_z_);
  }

 private:
  void ComputeLogPriorAndPrecCoeff(const Matrix<BaseFloat> &weight,
                                   const Vector<BaseFloat> &bias);

  void CompensateMultiFrameGmm(const Vector<double> &mu_h,
                               const Vector<double> &mu_z,
                               const Vector<double> &var_z, bool compensate_var,
                               int32 num_cepstral,
                               int32 num_fbank,
                               const Matrix<double> &dct_mat,
                               const Matrix<double> &inv_dct_mat,
                               int32 num_frames,
                               AmDiagGmm &noise_am_gmm);

  void ConvertToNNLayer(const AmDiagGmm &pos_am_gmm,
                        const AmDiagGmm &neg_am_gmm);

 private:
  Matrix<BaseFloat> cpu_linearity_;
  Vector<BaseFloat> cpu_bias_;

  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;

  Matrix<BaseFloat> cpu_linearity_corr_;
  Vector<BaseFloat> cpu_bias_corr_;

  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;

  Vector<double> log_prior_ratio_;  // positive to negative log prior ratio
  Matrix<double> precision_coeff_;  // interpolation weight for the positive precision matrix

  AmDiagGmm pos_am_gmm_, pos_noise_am_;
  AmDiagGmm neg_am_gmm_, neg_noise_am_;

  // parameters for VTS compensation
  int32 num_frame_;  // multiple frame input
  int32 delta_order_;  // how many deltas, currently only supports 2, i.e. static(0), delta(1) and acc(2)
  int32 num_cepstral_;
  int32 num_fbank_;
  BaseFloat ceplifter_;
  Matrix<double> dct_mat_;
  Matrix<double> inv_dct_mat_;

  // noise parameters
  bool compensate_var_;
  Vector<double> mu_h_;
  Vector<double> mu_z_;
  Vector<double> var_z_;
};

}  // namespace

#endif
