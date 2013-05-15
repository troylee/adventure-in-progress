/*
 * nnet/nnet-robm.cc
 *
 *  Created on: Apr 30, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "base/kaldi-math.h"
#include "nnet/nnet-rorbm.h"

namespace kaldi {

typedef kaldi::int32 int32;

void RoRbm::AddNoiseToData(CuMatrix<BaseFloat> &vt_cn) {
  int32 nVisNodes = vt_cn.NumCols();
  int32 nSide = (int32) (sqrt(nVisNodes));
  KALDI_ASSERT(nSide * nSide == nVisNodes);

  Matrix<BaseFloat> data_cpu(vt_cn.NumRows(), vt_cn.NumCols());
  vt_cn.CopyToMat(&data_cpu);

  Matrix<BaseFloat> noise_im(nSide, nSide);

  int32 offset, i_inds, j_inds;
  BaseFloat vvv;
  for (int32 n = 0; n < vt_cn.NumRows(); ++n) {
    if (RandUniform() > 0.5) {
      vvv = 0.1;
    } else {
      vvv = 0.9 * data_cpu.Max();
    }
    noise_im.SetRandn();
    noise_im.Scale(0.3);
    noise_im.Add(vvv);

    offset = RandInt(0, 2);

    for (int32 i = 0; i < nSide / 10; ++i) {
      for (int32 j = 0; j < nSide / 10; ++j) {

        for (int32 r = 0; r < 6; ++r) {
          for (int32 c = 0; c < 6; ++c) {
            i_inds = i * 10 + offset + r;
            j_inds = j * 10 + offset + c;
            data_cpu(n, j_inds * nSide + i_inds) = noise_im(i_inds, j_inds);
          }
        }
      }
    }

    for (int32 i = 0; i < vt_cn.NumCols(); ++i) {
      if (data_cpu(n, i) < 0)
        data_cpu(n, i) = 0.0;
      if (data_cpu(n, i) > 1)
        data_cpu(n, i) = 1.0;
    }

  }

  vt_cn.CopyFromMat(data_cpu);

}

/*
 * Assume vt_cn_ contains the current batch data.
 */
void RoRbm::NormalizeData(CuMatrix<BaseFloat> &vt_cn) {

  mat_tmp_.CopyFromMat(vt_cn);
  mat_tmp_.Power(2.0);  // data .* data
  vec_col_.AddRowSumMat(1.0, mat_tmp_, 0.0);  // sum(data .* data, 2)
  vec_col_.Power(0.5);  // sqrt(sum(data .* data, 2)) = datanorm
  vec_r_.CopyFromVec(vec_col_);  // datanorm
  vec_r_.Power(norm_k_);  // datanorm.^k
  vec_r_.Add(norm_eps_);  // eps + datanorm.^k
  vec_r_.InvertElements();  // 1.0/(eps + datanorm.^k)
  vec_r_.Add((BaseFloat) (-1.0 / (pow(norm_cc_, norm_k_) - norm_cc_)));  // 1.0/(eps + datanorm.^k) - 1.0/cc^k - cc
  vec_r_.DivElements(vec_col_);  // (1.0/(eps + datanorm.^k) - 1.0/cc^k - cc) ./ datanorm

  vec_col_.AddRowSumMat(1.0 / ((BaseFloat) vt_cn.NumCols()), vt_cn, 0.0);
  vt_cn.AddVecToCols(-1.0, vec_col_, 1.0);  // data - mean(data, 2)

  vt_cn.MulColsVec(vec_r_);  // data .* (1.0/(eps + datanorm.^k) - 1.0/cc^k - cc) ./ datanorm
  vt_cn.Scale(-1.0);  // data .* (-1.0/(eps + datanorm.^k) + 1.0/cc^k + cc) ./ datanorm
}

void RoRbm::ConvertNoiseHidBias(const CuVector<BaseFloat> &s_mu) {
  /* noise RBM hidden bias conversion, e = ee - s_mu * U */
  e_.CopyFromVec(ee_);  // ee
  U_tmp_.CopyFromMat(U_);  // U
  U_tmp_.MulColsVec(s_mu);  // s_mu * U
  e_.AddRowSumMat(-1.0, U_tmp_, 1.0);  // ee - s_mu * U
}

void RoRbm::InferChangeBatchSize(int32 bs) {
  if (bs < 0) {
    KALDI_ERR<< "Batch size much be greater than 0! [batchsize=" << bs << "].";
  }
  if(bs!=batch_size_) {
    vt_cn_.Resize(bs, vis_dim_);
    vt_cn_0_.Resize(bs, vis_dim_);

    mu_.Resize(bs, vis_dim_);
    mu_hat_.Resize(bs, vis_dim_);

    s_.Resize(bs, vis_dim_);
    phi_s_.Resize(bs, vis_dim_);

    log_sprob_0_.Resize(bs, vis_dim_);
    log_sprob_1_.Resize(bs, vis_dim_);

    mat_tmp_.Resize(bs, vis_dim_);

    v_condmean_.Resize(bs, vis_dim_);
    v_condstd_.Resize(bs, vis_dim_);

    ha_.Resize(bs, clean_hid_dim_);
    haprob_.Resize(bs, clean_hid_dim_);

    hs_.Resize(bs, noise_hid_dim_);
    hsprob_.Resize(bs, noise_hid_dim_);

    z_.Resize(bs, vis_dim_);

    vec_col_.Resize(bs);
    vec_r_.Resize(bs);
  }
}

void RoRbm::LearnChangeBatchSize(int32 bs) {
  if (bs < 0) {
    KALDI_ERR<< "Batch size much be greater than 0! [batchsize=" << bs << "].";
  }
  if(bs!=batch_size_) {
    InferChangeBatchSize(bs);

    fp_v_.Resize(bs, vis_dim_);
    fp_vt_.Resize(bs, vis_dim_);

    mu_t_hat_.Resize(bs, vis_dim_);

    fp_s_.Resize(bs, vis_dim_);

    fp_vt_condstd_.Resize(bs, vis_dim_);
    fp_vt_condmean_.Resize(bs, vis_dim_);

    fp_ha_.Resize(bs, clean_hid_dim_);

    fp_hs_.Resize(bs, noise_hid_dim_);

    batch_size_=bs;

  }
}

void RoRbm::InitializeInferVars() {

  std_hat_.Resize(vis_dim_);
  inv_gamma2_tmp_.Resize(vis_dim_);

  vec_tmp_.Resize(vis_dim_);

  InferChangeBatchSize(batch_size_);

}

void RoRbm::InitializeLearnVars() {

  InitializeInferVars();

  U_corr_.Resize(noise_hid_dim_, vis_dim_);
  d_corr_.Resize(vis_dim_);
  ee_corr_.Resize(noise_hid_dim_);
  bt_corr_.Resize(vis_dim_);
  lamt2_corr_.Resize(vis_dim_);
  gamma2_corr_.Resize(vis_dim_);

  U_pos_.Resize(noise_hid_dim_, vis_dim_);
  U_neg_.Resize(noise_hid_dim_, vis_dim_);
  d_pos_.Resize(vis_dim_);
  d_neg_.Resize(vis_dim_);
  ee_pos_.Resize(noise_hid_dim_);
  ee_neg_.Resize(noise_hid_dim_);
  bt_pos_.Resize(vis_dim_);
  bt_neg_.Resize(vis_dim_);
  lamt2_pos_.Resize(vis_dim_);
  lamt2_neg_.Resize(vis_dim_);
  gamma2_pos_.Resize(vis_dim_);
  gamma2_neg_.Resize(vis_dim_);

  U_tmp_.Resize(noise_hid_dim_, vis_dim_);

  e_.Resize(noise_hid_dim_);

  vec_tmp2_.Resize(vis_dim_);
  s_mu_.Resize(vis_dim_);
  lamt2_hat_.Resize(vis_dim_);

  LearnChangeBatchSize(batch_size_);

  U_corr_.SetZero();
  d_corr_.SetZero();
  ee_corr_.SetZero();
  bt_corr_.SetZero();
  lamt2_corr_.SetZero();
  gamma2_corr_.SetZero();

}

void RoRbm::ReadData(std::istream &is, bool binary) {

  /* Read in the layer types */
  std::string vis_node_type, clean_hid_node_type, noise_hid_node_type;
  ReadToken(is, binary, &vis_node_type);
  ReadToken(is, binary, &clean_hid_node_type);
  ReadToken(is, binary, &noise_hid_node_type);

  KALDI_ASSERT(vis_node_type == "gauss");
  KALDI_ASSERT(clean_hid_node_type == "bern");
  KALDI_ASSERT(noise_hid_node_type == "bern");

  /* Read in the hidden dim for noise RBM */
  ReadBasicType(is, binary, &noise_hid_dim_);
  KALDI_ASSERT(noise_hid_dim_ > 0);

  /* Read clean RBM */
  clean_vis_hid_.Read(is, binary);
  clean_vis_bias_.Read(is, binary);
  clean_hid_bias_.Read(is, binary);
  clean_vis_sigma_.Read(is, binary);

  KALDI_ASSERT(
      clean_vis_hid_.NumRows() == clean_hid_dim_ && clean_vis_hid_.NumCols() == vis_dim_);
  KALDI_ASSERT(clean_vis_bias_.Dim() == vis_dim_);
  KALDI_ASSERT(clean_hid_bias_.Dim() == clean_hid_dim_);
  KALDI_ASSERT(clean_vis_sigma_.Dim() == vis_dim_);

  clean_vis_sigma2_.CopyFromVec(clean_vis_sigma_);
  clean_vis_sigma2_.Power(2.0);

  /* Read Noise RBM */
  U_.Read(is, binary);  // weight matrix
  d_.Read(is, binary);  // visible bias
  ee_.Read(is, binary);  // hidden bias

  KALDI_ASSERT(U_.NumRows() == noise_hid_dim_ && U_.NumCols() == vis_dim_);
  KALDI_ASSERT(d_.Dim() == vis_dim_);
  KALDI_ASSERT(ee_.Dim() == noise_hid_dim_);

  /* Parameters for noisy inputs */
  bt_.Read(is, binary);
  lamt2_.Read(is, binary);
  gamma2_.Read(is, binary);

  KALDI_ASSERT(bt_.Dim() == vis_dim_);
  KALDI_ASSERT(lamt2_.Dim() == vis_dim_);
  KALDI_ASSERT(gamma2_.Dim() == vis_dim_);
}

void RoRbm::WriteData(std::ostream &os, bool binary) const {

  /* Write layer types */
  // vis type
  WriteToken(os, binary, "gauss");

  // clean hidden type
  WriteToken(os, binary, "bern");

  // noise hidden type
  WriteToken(os, binary, "bern");

  /* Write the hidden dim for noise RBM */
  WriteBasicType(os, binary, noise_hid_dim_);

  /* Write clean RBM */
  clean_vis_hid_.Write(os, binary);
  clean_vis_bias_.Write(os, binary);
  clean_hid_bias_.Write(os, binary);
  clean_vis_sigma_.Write(os, binary);

  /* Write noise RBM */
  U_.Write(os, binary);
  d_.Write(os, binary);
  ee_.Write(os, binary);

  /* Write noisy input parameters */
  bt_.Write(os, binary);
  lamt2_.Write(os, binary);
  gamma2_.Write(os, binary);
}

void RoRbm::PropagateFnc(const CuMatrix<BaseFloat> &vt_cn,
                         CuMatrix<BaseFloat> *v,
                         CuMatrix<BaseFloat> *v_condmean,
                         CuMatrix<BaseFloat> *ha,
                         CuMatrix<BaseFloat> *s,
                         CuMatrix<BaseFloat> *hs) {

  int32 n = vt_cn.NumRows();
  if (n != batch_size_) {
    /* Resize the necessary variables */
    LearnChangeBatchSize(n);
    v_condmean->Resize(n, vis_dim_);
    ha->Resize(n, clean_hid_dim_);
    s->Resize(n, vis_dim_);
    hs->Resize(n, vis_dim_);
  }

  /* initialize the clean RBM hidden states */
  haprob_.AddVecToRows(1.0, clean_hid_bias_, 0.0);  // c
  haprob_.AddMatMat(1.0, vt_cn, kNoTrans, clean_vis_hid_, kTrans, 1.0);
  cu::Sigmoid(haprob_, &haprob_);
  cu_rand_.BinarizeProbs(haprob_, ha);

  /* initialize the noise RBM hidden states */
  cu_rand_.RandUniform(hs);

  /* do inference */
  z_.SetZero();

  /* run multiple iterations to denoise */
  for (int32 k = 0; k < num_infer_iters_; ++k) {
    // downsample - from hidden to visible
    /* needed for sprob_0, clean GRBM */
    mu_.AddMatMat(1.0, *ha, kNoTrans, clean_vis_hid_, kNoTrans, 0.0);  // ha * W
    mu_.MulColsVec(clean_vis_var_);  // var * (ha * W)
    mu_.AddVecToRows(1.0, clean_vis_bias_, 1.0);  // b + var * (ha * W)
    /* needed for sprob_1, noise RBM */
    phi_s_.AddVecToRows(1.0, d_, 0.0);  // d
    phi_s_.AddMatMat(1.0, *hs, kNoTrans, U_, kNoTrans, 1.0);  // d + hs * U

    /* needed for sprob_1, noisy input */
    mu_hat_.CopyFromMat(vt_cn);
    mu_hat_.MulColsVec(gamma2_);  // gamma2 .* vt_cn
    mu_hat_.AddMat(1.0, mu_, 1.0);  // mu + gamma2 .* vt_cn
    vec_tmp_.CopyFromVec(gamma2_);  // gamma2
    vec_tmp_.Add(1.0);  // gamma2 + 1
    mu_hat_.DivColsVec(vec_tmp_);  // (mu + gamma2 .* vt_cn) ./ (gamma2 + 1)

    /* needed for sprob_1 */
    vec_tmp_.Power(0.5);  // sqrt(gamma2 + 1)
    std_hat_.CopyFromVec(clean_vis_std_);  // std_vec
    std_hat_.DivElements(vec_tmp_);  // std_vec ./ sqrt(gamma2 + 1)

    /* compute log_sprob_1 */
    log_sprob_1_.CopyFromMat(phi_s_);  // phi_s

    mat_tmp_.CopyFromMat(vt_cn);  // vt_cn
    mat_tmp_.Power(2.0);  // vt_cn.^2
    vec_tmp_.CopyFromVec(gamma2_);  // gamma2
    vec_tmp_.DivElements(clean_vis_var_);  // gamma2 ./ var_vec
    mat_tmp_.MulColsVec(vec_tmp_);  // vt_cn.^2 .* gamma2 ./ var_vec
    log_sprob_1_.AddMat(-0.5, mat_tmp_, 1.0);  // phi_s - 0.5 * vt_cn.^2 .* gamma2 ./ var_vec

    mat_tmp_.CopyFromMat(mu_hat_);  // mu_hat
    mat_tmp_.DivColsVec(std_hat_);  // mu_hat ./ std_hat
    mat_tmp_.Power(2.0);  // mu_hat.^2 ./ std_hat.^2
    log_sprob_1_.AddMat(0.5, mat_tmp_, 1.0);  // phi_s - 0.5 * vt_cn.^2 .* gamma2 ./ var_vec + 0.5 * mu_hat.^2 ./ std_hat.^2

    vec_tmp_.CopyFromVec(std_hat_);  // std_hat
    vec_tmp_.ApplyLog();  // log(std_hat)
    log_sprob_1_.AddVecToRows(1.0, vec_tmp_, 1.0);

    /* compute log_sprob_0 */
    mat_tmp_.CopyFromMat(mu_);  // mu
    mat_tmp_.Power(2.0);  // mu.^2
    mat_tmp_.DivColsVec(clean_vis_var_);  // mu.^2 ./ var_vec
    log_sprob_0_.AddMat(0.5, mat_tmp_, 0.0);  // mu.^2 ./ var_vec

    vec_tmp_.CopyFromVec(clean_vis_std_);  // std_vec
    vec_tmp_.ApplyLog();  // log(std_vec)
    log_sprob_0_.AddVecToRows(1.0, vec_tmp_, 1.0);  // mu.^2 ./ var_vec + log(std_vec)

    /* log(exp(log_sprob_0) + exp(log_sprob_1)) */
    log_sprob_0_.LogAddExpMat(log_sprob_1_);

    /* compute sprob (saved in log_sprob_1) */
    log_sprob_1_.AddMat(-1.0, log_sprob_0_);  // log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1))
    log_sprob_1_.ApplyExp();  // exp(log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1)))

    /* compute s */
    cu_rand_.BinarizeProbs(log_sprob_1_, s);

    /* compute v_condmean */
    v_condmean->CopyFromMat(mu_);  // mu

    mat_tmp_.CopyFromMat(vt_cn);  // vt_cn
    mat_tmp_.MulElements(*s);  // s .* vt_cn
    mat_tmp_.MulColsVec(gamma2_);  // gamma2 .* s.* vt_cn
    v_condmean->AddMat(1.0, mat_tmp_, 1.0);  // gamma2 .* s.* vt_cn + mu

    mat_tmp_.CopyFromMat(*s);  // s
    mat_tmp_.MulColsVec(gamma2_);  // gamma2 .* s
    mat_tmp_.Add(1.0);  // gamma2 .* s + 1
    v_condmean->DivElements(mat_tmp_);  // (gamma2 .* s.* vt_cn + mu) ./ (gamma2 .* s + 1)

    /* compute v_condstd */
    v_condstd_.AddVecToRows(1.0, clean_vis_std_, 0.0);  // std_vec
    mat_tmp_.Power(0.5);  // sqrt(gamma2 .* s + 1)
    v_condstd_.DivElements(mat_tmp_);  // std_vec ./ sqrt(gamma2 .* s + 1)

    /* sample from v */
    cu_rand_.RandGaussian(v);
    v->MulElements(v_condstd_);
    v->AddMat(1.0, v_condmean_, 1.0);

    //TODO::check the correctness
    /* normalise the masked vt_cn */
    vt_cn_s_.CopyFromMat(vt_cn);
    vt_cn_s_.MulElements(*s);  // vt_cn .* s, when s==1, it is uncorrupted
    NormalizeData(vt_cn_s_);  // ncc_func(vt_cn .* s), normalize only on the uncorrupted data
    vt_cn_s_.MulElements(s_);  // keep the normalized uncorrupted data
    mat_tmp_.CopyFromMat(s_);
    mat_tmp_.Add(-1.0);  // change 0 to -1 and 1 to 0
    mat_tmp_.MulElements(vt_cn_s_);  // noise components
    vt_cn_s_.AddMat(-1.0, mat_tmp_, 1.0);  // Now for vt_cn, is clean speech when s==1; is noise when s==0

    /* sample the hidden variables */
    haprob_.AddVecToRows(1.0, clean_hid_bias_, 0.0);  // c
    haprob_.AddMatMat(1.0, v, kNoTrans, clean_vis_hid_, kTrans, 1.0);  // v*W + c
    cu::Sigmoid(haprob_, &haprob_);  // 1.0 ./ (1.0 + exp(v*W + c))
    cu_rand_.BinarizeProbs(haprob_, &ha_);  // binarize

    hsprob_.AddVecToRows(1.0, e_, 0.0);  // e
    hsprob_.AddMatMat(1.0, s_, kNoTrans, U_, kTrans, 1.0);  // s*U + e
    cu::Sigmoid(hsprob_, &hsprob_);  // 1.0 ./ (1.0 + exp(s*U + e))
    cu_rand_.BinarizeProbs(hsprob_, &hs_);  // binarize

    /* collect smooth estimates */
    if (z_start_iter_ >= 0) {  // negative z indicates no collection
      if (k == z_start_iter_) {
        z_.CopyFromMat(v_condmean_);
      } else if (k > z_start_iter_) {
        z_.AddMat(1 - z_momentum_, v_condmean_, z_momentum_);
      }
    }

  }  // end iteration k

}

/*
 * Requires: W, b, c, U, d, e, gamma2, std_vec, vt_cn, ha, hs, ncc_func, params
 */
void RoRbm::Infer(CuMatrix<BaseFloat> &v) {

  vt_cn_0_.CopyFromMat(vt_cn_);  // save for future use
  z_.SetZero();

  /* run multiple iterations to denoise */
  for (int32 k = 0; k < num_pos_iters_; ++k) {
    // downsample - from hidden to visible
    /* needed for sprob_0, clean GRBM */
    mu_.AddMatMat(1.0, ha_, kNoTrans, clean_vis_hid_, kNoTrans, 0.0);  // ha * W
    mu_.MulColsVec(clean_vis_sigma2_);  // sigma2 * (ha * W)
    mu_.AddVecToRows(1.0, clean_vis_bias_, 1.0);  // b + sigma2 * (ha * W)
    /* needed for sprob_1, noise RBM */
    phi_s_.AddVecToRows(1.0, d_, 0.0);  // d
    phi_s_.AddMatMat(1.0, hs_, kNoTrans, U_, kNoTrans, 1.0);  // d + hs * U

    /* needed for sprob_1, noisy input */
    mu_hat_.CopyFromMat(vt_cn_);
    mu_hat_.MulColsVec(gamma2_);  // gamma2 .* vt_cn
    mu_hat_.AddMat(1.0, mu_, 1.0);  // mu + gamma2 .* vt_cn
    vec_tmp_.CopyFromVec(gamma2_);  // gamma2
    vec_tmp_.Add(1.0);  // gamma2 + 1
    mu_hat_.DivColsVec(vec_tmp_);  // (mu + gamma2 .* vt_cn) ./ (gamma2 + 1)

    /* needed for sprob_1 */
    vec_tmp_.Power(0.5);  // sqrt(gamma2 + 1)
    std_hat_.CopyFromVec(clean_vis_sigma_);  // std_vec
    std_hat_.DivElements(vec_tmp_);  // std_vec ./ sqrt(gamma2 + 1)

    /* compute log_sprob_1 */
    log_sprob_1_.CopyFromMat(phi_s_);  // phi_s

    mat_tmp_.CopyFromMat(vt_cn_);  // vt_cn
    mat_tmp_.Power(2.0);  // vt_cn.^2
    vec_tmp_.CopyFromVec(gamma2_);  // gamma2
    vec_tmp_.DivElements(clean_vis_sigma2_);  // gamma2 ./ var_vec
    mat_tmp_.MulColsVec(vec_tmp_);  // vt_cn.^2 .* gamma2 ./ var_vec
    log_sprob_1_.AddMat(-0.5, mat_tmp_, 1.0);  // phi_s - 0.5 * vt_cn.^2 .* gamma2 ./ var_vec

    mat_tmp_.CopyFromMat(mu_hat_);  // mu_hat
    mat_tmp_.DivColsVec(std_hat_);  // mu_hat ./ std_hat
    mat_tmp_.Power(2.0);  // mu_hat.^2 ./ std_hat.^2
    log_sprob_1_.AddMat(0.5, mat_tmp_, 1.0);  // phi_s - 0.5 * vt_cn.^2 .* gamma2 ./ var_vec + 0.5 * mu_hat.^2 ./ std_hat.^2

    vec_tmp_.CopyFromVec(std_hat_);  // std_hat
    vec_tmp_.ApplyLog();  // log(std_hat)
    log_sprob_1_.AddVecToRows(1.0, vec_tmp_, 1.0);

    /* compute log_sprob_0 */
    mat_tmp_.CopyFromMat(mu_);  // mu
    mat_tmp_.Power(2.0);  // mu.^2
    mat_tmp_.DivColsVec(clean_vis_sigma2_);  // mu.^2 ./ var_vec
    log_sprob_0_.AddMat(0.5, mat_tmp_, 0.0);  // mu.^2 ./ var_vec

    vec_tmp_.CopyFromVec(clean_vis_sigma_);  // std_vec
    vec_tmp_.ApplyLog();  // log(std_vec)
    log_sprob_0_.AddVecToRows(1.0, vec_tmp_, 1.0);  // mu.^2 ./ var_vec + log(std_vec)

    /* log(exp(log_sprob_0) + exp(log_sprob_1)) */
    log_sprob_0_.LogAddExpMat(log_sprob_1_);

    /* compute sprob (saved in log_sprob_1) */
    log_sprob_1_.AddMat(-1.0, log_sprob_0_);  // log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1))
    log_sprob_1_.ApplyExp();  // exp(log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1)))

    /* compute s */
    cu_rand_.BinarizeProbs(log_sprob_1_, &s_);

    /* compute v_condmean */
    v_condmean_.CopyFromMat(mu_);  // mu

    mat_tmp_.CopyFromMat(vt_cn_);  // vt_cn
    mat_tmp_.MulElements(s_);  // s .* vt_cn
    mat_tmp_.MulColsVec(gamma2_);  // gamma2 .* s.* vt_cn
    v_condmean_.AddMat(1.0, mat_tmp_, 1.0);  // gamma2 .* s.* vt_cn + mu

    mat_tmp_.CopyFromMat(s_);  // s
    mat_tmp_.MulColsVec(gamma2_);  // gamma2 .* s
    mat_tmp_.Add(1.0);  // gamma2 .* s + 1
    v_condmean_.DivElements(mat_tmp_);  // (gamma2 .* s.* vt_cn + mu) ./ (gamma2 .* s + 1)

    /* compute v_condstd */
    v_condstd_.AddVecToRows(1.0, clean_vis_sigma_, 0.0);  // std_vec
    mat_tmp_.Power(0.5);  // sqrt(gamma2 .* s + 1)
    v_condstd_.DivElements(mat_tmp_);  // std_vec ./ sqrt(gamma2 .* s + 1)

    /* sample from v */
    cu_rand_.RandGaussian(&v);
    v.MulElements(v_condstd_);
    v.AddMat(1.0, v_condmean_, 1.0);

    /* normalise the masked vt_cn */
    vt_cn_.MulElements(s_);  // vt_cn .* s, when s==1, it is uncorrupted
    NormalizeBatchData();  // ncc_func(vt_cn .* s), normalize only on the uncorrupted data
    vt_cn_.MulElements(s_);  // keep the normalized uncorrupted data
    mat_tmp_.CopyFromMat(s_);
    mat_tmp_.Add(-1.0);  // change 0 to -1 and 1 to 0
    mat_tmp_.MulElements(vt_cn_0_);  // noise components
    vt_cn_.AddMat(-1.0, mat_tmp_, 1.0);  // Now for vt_cn, is clean speech when s==1; is noise when s==0

    /* sample the hidden variables */
    haprob_.AddVecToRows(1.0, clean_hid_bias_, 0.0);  // c
    haprob_.AddMatMat(1.0, v, kNoTrans, clean_vis_hid_, kTrans, 1.0);  // v*W + c
    cu::Sigmoid(haprob_, &haprob_);  // 1.0 ./ (1.0 + exp(v*W + c))
    cu_rand_.BinarizeProbs(haprob_, &ha_);  // binarize

    hsprob_.AddVecToRows(1.0, e_, 0.0);  // e
    hsprob_.AddMatMat(1.0, s_, kNoTrans, U_, kTrans, 1.0);  // s*U + e
    cu::Sigmoid(hsprob_, &hsprob_);  // 1.0 ./ (1.0 + exp(s*U + e))
    cu_rand_.BinarizeProbs(hsprob_, &hs_);  // binarize

    /* collect smooth estimates */
    if (z_start_iter_ >= 0) {  // negative z indicates no collection
      if (k == z_start_iter_) {
        z_.CopyFromMat(v_condmean_);
      } else if (k > z_start_iter_) {
        z_.AddMat(1 - z_momentum_, v_condmean_, z_momentum_);
      }
    }

  }  // end iteration k
}  // end Infer()

/*
 * vt_cn: the noisy inputs
 * v: the clean version of the inputs
 */
void RoRbm::Learn(const CuMatrix<BaseFloat> &vt, CuMatrix<BaseFloat> &v) {

  if (v.NumRows() != vt.NumRows() || v.NumCols() != vt.NumCols()) {
    v.Resize(vt.NumRows(), vt.NumCols());
  }

  int32 n = vt.NumRows();
  if (n != batch_size_) {
    /* Resize the necessary variables */
    LearnChangeBatchSize(n);
  }

  vt_cn_.CopyFromMat(vt);
  /* add noise to the training data */
  AddNoiseToBatchData();
  /* normalize the data */
  NormalizeBatchData();

  /*
   * Initialize the fantasy particles at the first bunch of data.
   */
  if (first_data_bunch_) {
    cu_rand_.RandUniform(&fp_ha_);
    cu_rand_.RandUniform(&fp_hs_);
    fp_vt_.CopyFromMat(vt_cn_);

    s_mu_.Set(0.9);  // moving average of the mean of the layer s

    first_data_bunch_ = false;
  }

  /* noise RBM hidden bias conversion, e = ee - s_mu * U */
  e_.CopyFromVec(ee_);  // ee
  U_tmp_.CopyFromMat(U_);  // U
  U_tmp_.MulColsVec(s_mu_);  // s_mu * U
  e_.AddRowSumMat(-1.0, U_tmp_, 1.0);  // ee - s_mu * U

  /* initialize the clean RBM hidden states */
  haprob_.AddVecToRows(1.0, clean_hid_bias_, 0.0);  // c
  haprob_.AddMatMat(1.0, vt_cn_, kNoTrans, clean_vis_hid_, kTrans, 1.0);
  cu::Sigmoid(haprob_, &haprob_);
  cu_rand_.BinarizeProbs(haprob_, &ha_);

  /* initialize the noise RBM hidden states */
  cu_rand_.RandUniform(&hs_);

  Infer(v);

  /* use more smoother version */
  v.CopyFromMat(v_condmean_);

  s_mu_.AddColSumMat(0.05, s_, 0.95);

  /* positive phase gradient */
  mat_tmp_.CopyFromMat(vt_cn_);  // vt_cn
  mat_tmp_.MulColsVec(lamt2_);  // vt_cn .* lamt2
  bt_pos_.AddColSumMat(1.0, mat_tmp_, 0.0);  // sum(vt_cn .* lamt2)

  mat_tmp_.AddVecToRows(1.0, bt_, 0.0);  // bt
  mat_tmp_.AddMat(-0.5, vt_cn_, 1.0);  // -0.5 * vt_cn + bt
  mat_tmp_.MulElements(vt_cn_);  // -0.5 * vt_cn.*vt_cn + vt_cn .* bt
  lamt2_pos_.AddColSumMat(1.0, mat_tmp_, 0.0);  // sum(-0.5 * vt_cn.^2 + vt_cn .* bt)

  mat_tmp_.CopyFromMat(vt_cn_);  // vt_cn
  mat_tmp_.AddMat(1.0, v, -1.0);  // v - vt_cn
  mat_tmp_.Power(2.0);  // (v - vt_cn).^2
  mat_tmp_.MulElements(s_);  // s .* (v - vt_cn).^2
  mat_tmp_.Scale(-0.5);  // -0.5 * s.* (v - vt_cn).^2
  gamma2_pos_.AddColSumMat(1.0, mat_tmp_, 0.0);  // sum(-0.5 * s.* (v - vt_cn).^2)
  gamma2_pos_.DivElements(clean_vis_sigma2_);  // sum(-0.5 * s.* (v - vt_cn).^2) ./ var_vec

  mat_tmp_.CopyFromMat(s_);  // s
  mat_tmp_.AddVecToRows(-1.0, s_mu_, 1.0);  // s - s_mu
  U_pos_.AddMatMat(1.0, hs_, kTrans, mat_tmp_, kNoTrans, 0.0);
  d_pos_.AddColSumMat(1.0, mat_tmp_, 0.0);
  ee_pos_.AddColSumMat(1.0, hs_, 0.0);

  /* update using SAP */
  for (int32 kk = 0; kk < num_gibbs_iters_; ++kk) {

    /* #1. p(s|hs, ha, vt) */
    mu_.AddMatMat(1.0, fp_ha_, kNoTrans, clean_vis_hid_, kNoTrans, 0.0);  // fp_ha * W
    mu_.MulColsVec(clean_vis_sigma2_);  // (fp_ha * W) .* var_vec
    mu_.AddVecToRows(1.0, clean_vis_bias_, 1.0);  // (fp_ha * W) .* var_vec + b

    phi_s_.AddVecToRows(1.0, d_, 0.0);  // d
    phi_s_.AddMatMat(1.0, fp_hs_, kNoTrans, U_, kNoTrans, 1.0);  // fp_hs * U + d

    mu_hat_.CopyFromMat(fp_vt_);  // fp_vt
    mu_hat_.MulColsVec(gamma2_);  // fp_vt .* gamma2
    mu_hat_.AddMat(1.0, mu_, 1.0);  // mu + fp_vt .* gamma2
    vec_tmp_.CopyFromVec(gamma2_);  // gamma2
    vec_tmp_.Add(1.0);  // gamma2 + 1
    vec_tmp_.InvertElements();  // 1.0 / (gamma2 + 1)
    mu_hat_.MulColsVec(vec_tmp_);  // (mu + fp_vt .* gamma2) ./ (gamma2 + 1)

    std_hat_.CopyFromVec(clean_vis_sigma_);  // std_vec
    vec_tmp_.Power(0.5);  // 1.0 / sqrt(gamma2 + 1)
    std_hat_.DivElements(vec_tmp_);  // std_vec ./ sqrt(gamma2 + 1)

    /** compute log_sprob_1 **/
    log_sprob_1_.CopyFromMat(phi_s_);  // phi_s

    mat_tmp_.CopyFromMat(fp_vt_);  // fp_vt
    mat_tmp_.Power(2);  // fp_vt.^2
    mat_tmp_.MulColsVec(gamma2_);  // gamma2 .* (fp_vt.^2)
    mat_tmp_.DivColsVec(clean_vis_sigma2_);  // gamma2 .* (fp_vt.^2) ./ var_vec
    log_sprob_1_.AddMat(-0.5, mat_tmp_, 1.0);  // phi_s - 0.5 * gamma2 .* (fp_vt.^2) ./ var_vec

    mat_tmp_.CopyFromMat(mu_hat_);  // mu_hat
    mat_tmp_.DivColsVec(std_hat_);  // mu_hat ./ std_hat
    mat_tmp_.Power(2.0);  // mu_hat.^2 ./ std_hat.^2
    log_sprob_1_.AddMat(0.5, mat_tmp_, 1.0);  // phi_s - 0.5 * gamma2 .* (fp_vt.^2) ./ var_vec + 0.5 * mu_hat.^2 ./ std_hat.^2

    vec_tmp_.CopyFromVec(std_hat_);  // std_hat
    vec_tmp_.ApplyLog();  // log(std_hat)
    log_sprob_1_.AddVecToRows(1.0, vec_tmp_, 1.0);  //  phi_s - 0.5 * gamma2 .* (fp_vt.^2) ./ var_vec + 0.5 * mu_hat.^2 ./ std_hat.^2 + log(std_hat)

    /** compute log_sprob_0 **/
    log_sprob_0_.CopyFromMat(mu_);  // mu
    log_sprob_0_.Power(2.0);  // mu.^2
    log_sprob_0_.DivColsVec(clean_vis_sigma2_);  // mu.^2 ./ var_vec
    log_sprob_0_.Scale(0.5);  // 0.5 * mu.^2 ./ var_vec
    vec_tmp_.CopyFromVec(clean_vis_sigma_);  // std_vec
    vec_tmp_.ApplyLog();  // log(std_vec)
    log_sprob_0_.AddVecToRows(1.0, vec_tmp_, 1.0);

    /** log(exp(log_sprob_0) + exp(log_sprob_1)) **/
    log_sprob_0_.LogAddExpMat(log_sprob_1_);

    /** compute sprob (saved in log_sprob_1) **/
    log_sprob_1_.AddMat(-1.0, log_sprob_0_);  // log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1))
    log_sprob_1_.ApplyExp();  // exp(log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1)))

    /** compute s **/
    cu_rand_.BinarizeProbs(log_sprob_1_, &fp_s_);

    /* #2. p(v|s, ha, vt) */
    mat_tmp_.CopyFromMat(fp_s_);  // fp_s
    mat_tmp_.MulColsVec(gamma2_);  // gamma2 .* fp_s
    v_condmean_.CopyFromMat(mat_tmp_);  // gamma2 .* fp_s
    v_condmean_.MulElements(fp_vt_);  // gamma2 .* fp_s .* fp_vt
    v_condmean_.AddMat(1.0, mu_, 1.0);  // gamma2 .* fp_s .* fp_vt + mu
    mat_tmp_.Add(1.0);  // gamma2 .* fp_s + 1.0
    v_condmean_.DivElements(mat_tmp_);  // (gamma2 .* fp_s .* fp_vt + mu) ./ (gamma2 .* fp_s + 1.0)

    v_condstd_.CopyFromMat(mat_tmp_);  // gamma2 .* fp_s + 1.0
    v_condstd_.Power(0.5);  // sqrt(gamma2 .* fp_s + 1.0)
    v_condstd_.InvertElements();  // 1.0 ./ sqrt(gamma2 .* fp_s + 1.0)
    v_condstd_.MulColsVec(clean_vis_sigma_);  // std_vec ./ sqrt(gamma2 .* fp_s + 1.0)

    /** sample from v **/
    cu_rand_.RandGaussian(&fp_v_);  // random
    fp_v_.MulElements(v_condstd_);  // fp_v .* v_condstd
    fp_v_.AddMat(1.0, v_condmean_, 1.0);  // fp_v .* v_condstd + v_condmean

    /* #3. p(s|v, hs) */
    vec_tmp_.CopyFromVec(clean_vis_sigma2_);  // var_vec
    vec_tmp_.MulElements(bt_);  // var_vec .* bt
    vec_tmp2_.CopyFromVec(gamma2_);  // gamma2
    vec_tmp2_.DivElements(lamt2_);  // gamma2 ./ lamt2
    mu_t_hat_.CopyFromMat(fp_v_);  // fp_v
    mu_t_hat_.MulColsVec(vec_tmp2_);  // (gamma2 ./ lamt2) .* fp_v
    mu_t_hat_.AddVecToRows(1.0, vec_tmp_, 1.0);  // var_vec .* bt + (gamma2 ./ lamt2) .* fp_v
    vec_tmp2_.AddVec(1.0, clean_vis_sigma2_, 1.0);  // var_vec + gamma2 ./ lamt2
    mu_t_hat_.DivColsVec(vec_tmp2_);  // (var_vec .* bt + (gamma2 ./ lamt2) .* fp_v) ./ (var_vec + gamma2 ./ lamt2)

    lamt2_hat_.CopyFromVec(vec_tmp2_);  // var_vec + gamma2 ./ lamt2
    lamt2_hat_.DivElements(clean_vis_sigma2_);  // (var_vec + gamma2 ./ lamt2) ./ var_vec
    lamt2_hat_.MulElements(lamt2_);  // (var_vec + gamma2 ./ lamt2) ./ (var_vec ./ lamt2)

    /** compute log_sprob_1 **/
    log_sprob_1_.CopyFromMat(phi_s_);  // phi_s

    mat_tmp_.CopyFromMat(fp_v_);  // fp_v
    mat_tmp_.Power(2);  // fp_v.^2
    mat_tmp_.MulColsVec(gamma2_);  // gamma2 .* (fp_v.^2)
    mat_tmp_.DivColsVec(clean_vis_sigma2_);  // gamma2 .* (fp_v.^2) ./ var_vec
    log_sprob_1_.AddMat(-0.5, mat_tmp_, 1.0);  // phi_s - 0.5 * gamma2 .* (fp_v.^2) ./ var_vec

    mat_tmp_.CopyFromMat(mu_t_hat_);  // mu_t_hat
    mat_tmp_.Power(2.0);  // mu_t_hat.^2
    mat_tmp_.MulColsVec(lamt2_hat_);  // (mu_t_hat.^2) .* lamt2_hat
    log_sprob_1_.AddMat(0.5, mat_tmp_, 1.0);  // phi_s - 0.5 * gamma2 .* (fp_v.^2) ./ var_vec + 0.5 * mu_t_hat.^2 .* lamt2_hat

    vec_tmp_.CopyFromVec(lamt2_hat_);  // lamt2_hat
    vec_tmp_.ApplyLog();  // log(lamt2_hat)
    log_sprob_1_.AddVecToRows(-0.5, vec_tmp_, 1.0);  //  phi_s - 0.5 * gamma2 .* (fp_v.^2) ./ var_vec + 0.5 * mu_t_hat.^2 .* lamt2_hat - log(sqrt(lamt2_hat))

    /** compute log_sprob_0 **/
    vec_tmp_.CopyFromVec(bt_);  // bt
    vec_tmp_.Power(2.0);  // bt.^2
    vec_tmp_.MulElements(lamt2_);  // bt.^2 .* lamt2
    vec_tmp2_.CopyFromVec(lamt2_);  // lamt2
    vec_tmp2_.ApplyLog();  // log(lamt2)
    vec_tmp_.AddVec(-0.5, vec_tmp2_, 0.5);  // 0.5 * bt.^2 .* lamt2 - log(sqrt(lmat2))
    log_sprob_0_.AddVecToRows(1.0, vec_tmp_, 0.0);

    /** log(exp(log_sprob_0) + exp(log_sprob_1)) **/
    log_sprob_0_.LogAddExpMat(log_sprob_1_);

    /** compute sprob (saved in log_sprob_1) **/
    log_sprob_1_.AddMat(-1.0, log_sprob_0_);  // log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1))
    log_sprob_1_.ApplyExp();  // exp(log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1)))

    cu_rand_.BinarizeProbs(log_sprob_1_, &fp_s_);

    /* #4. p(vt | s, v) */
    vec_tmp_.CopyFromVec(gamma2_);  // gamma2
    vec_tmp_.DivElements(lamt2_);  // gamma2 ./ lamt2
    mat_tmp_.CopyFromMat(fp_s_);  // fp_s
    mat_tmp_.MulColsVec(vec_tmp_);  // fp_s .* (gamma2 ./ lamt2)
    vec_tmp_.CopyFromVec(clean_vis_sigma2_);  // var_vec
    vec_tmp_.MulElements(bt_);  // var_vec .* bt
    fp_vt_condmean_.CopyFromMat(mat_tmp_);  // fp_s .* (gamma2 ./ lamt2)
    fp_vt_condmean_.MulElements(fp_v_);  // fp_s .* (gamma2 ./ lamt2) .* fp_v
    fp_vt_condmean_.AddVecToRows(1.0, vec_tmp_, 1.0);  // var_vec .* bt + fp_s .* (gamma2 ./ lamt2) .* fp_v
    mat_tmp_.AddVecToRows(1.0, clean_vis_sigma2_, 1.0);  // var_vec + fp_s .* (gamma2 ./ lamt2)
    fp_vt_condmean_.DivElements(mat_tmp_);  // (var_vec .* bt + fp_s .* (gamma2 ./ lamt2) .* fp_v) ./ (var_vec + fp_s .* (gamma2 ./ lamt2))

    vec_tmp_.CopyFromVec(clean_vis_sigma2_);  // var_vec
    vec_tmp_.DivElements(lamt2_);  // var_vec ./ lamt2
    fp_vt_condstd_.AddVecToRows(1.0, vec_tmp_, 0.0);  // var_vec ./ lamt2
    fp_vt_condstd_.DivElements(mat_tmp_);  // (var_vec ./ lamt2) ./ (var_vec + fp_s .* (gamma2 ./ lamt2))
    fp_vt_condstd_.Power(0.5);  // sqrt((var_vec ./ lamt2) ./ (var_vec + fp_s .* (gamma2 ./ lamt2)))

    /** sample from vt **/
    cu_rand_.RandGaussian(&fp_vt_);
    fp_vt_.MulElements(fp_vt_condstd_);  // fp_vt .* fp_vt_condstd
    fp_vt_.AddMat(1.0, fp_vt_condmean_, 1.0);  // fp_vt .* fp_vt_condstd + fp_vt_condmean

    /* #5. p(hs|s); p(ha|v) */
    haprob_.AddVecToRows(1.0, clean_hid_bias_, 0.0);  // c
    haprob_.AddMatMat(1.0, fp_v_, kNoTrans, clean_vis_hid_, kTrans, 1.0);  // fp_v * W' + c
    cu::Sigmoid(haprob_, &haprob_);
    cu_rand_.BinarizeProbs(haprob_, &fp_ha_);

    hsprob_.AddVecToRows(1.0, e_, 0.0);  // e
    hsprob_.AddMatMat(1.0, fp_s_, kNoTrans, U_, kTrans, 1.0);  // fp_s * U' + e
    cu::Sigmoid(hsprob_, &hsprob_);
    cu_rand_.BinarizeProbs(hsprob_, &fp_hs_);

  }  // iteration kk, end SAP

  /* save temporal results (vt_cn, v, s, fp_s) here if needed */

  /* negative phase gradients */
  mat_tmp_.CopyFromMat(fp_vt_);  // fp_vt
  mat_tmp_.MulColsVec(lamt2_);  // fp_vt .* lamt2
  bt_neg_.AddColSumMat(1.0, mat_tmp_, 0.0);

  mat_tmp_.AddVecToRows(1.0, bt_, 0.0);  // bt
  mat_tmp_.AddMat(-0.5, fp_vt_, 1.0);  // -0.5 * fp_vt + bt
  mat_tmp_.MulElements(fp_vt_);  // -0.5 * fp_vt.^2 + fp_vt .* bt
  lamt2_neg_.AddColSumMat(1.0, mat_tmp_, 0.0);

  mat_tmp_.CopyFromMat(fp_v_);  // fp_v
  mat_tmp_.AddMat(-1.0, fp_vt_, 1.0);  // fp_v - fp_vt
  mat_tmp_.Power(2.0);  // (fp_v - fp_vt).^2
  mat_tmp_.MulElements(fp_s_);  // fp_s .* (fp_v - fp_vt).^2
  mat_tmp_.DivColsVec(clean_vis_sigma2_);  // fp_s .* (fp_v - fp_vt).^2 ./ var_vec
  gamma2_neg_.AddColSumMat(-0.5, mat_tmp_, 0.0);  // -0.5 * fp_s .* (fp_v - fp_vt).^2 ./ var_vec

  mat_tmp_.CopyFromMat(fp_s_);  // fp_s
  mat_tmp_.AddVecToRows(-1.0, s_mu_, 1.0);  // fp_s - s_mu
  U_neg_.AddMatMat(1.0, fp_hs_, kTrans, mat_tmp_, kNoTrans, 0.0);  // (fp_s - s_mu)' * fp_hs
  d_neg_.AddColSumMat(1.0, mat_tmp_, 0.0);
  ee_neg_.AddColSumMat(1.0, fp_hs_, 0.0);

  ////////////////////////////////////////////////////////////////

  BaseFloat lr = learn_rate_ / n;
  BaseFloat wc = -learn_rate_ * l2_penalty_;

  bt_pos_.AddVec(-1.0, bt_neg_, 1.0);  // bt_pos - bt_neg
  bt_corr_.AddVec(lr, bt_pos_, momentum_);  // momentum * bt_inc + epsilon/n * (bt_pos - bt_neg)
  bt_corr_.AddVec(wc, bt_, 1.0);  // momentum * bt_inc + epsilon/n * (bt_pos - bt_neg) - epsilon * wtcost * bt

  lamt2_pos_.AddVec(-1.0, lamt2_neg_, 1.0);  // lamt2_pos - lamt2_neg
  lamt2_corr_.AddVec(lr, lamt2_pos_, momentum_);
  lamt2_corr_.AddVec(wc, lamt2_, 1.0);

  gamma2_pos_.AddVec(-1.0, gamma2_neg_, 1.0);
  gamma2_corr_.AddVec(0.1 * lr, gamma2_pos_, momentum_);  // gamma2 has relative small learn rate
  gamma2_corr_.AddVec(0.1 * wc, gamma2_, 1.0);  // gamma2 has relative small learn rate

  d_pos_.AddVec(-1.0, d_neg_, 1.0);
  d_corr_.AddVec(lr, d_pos_, momentum_);

  ee_pos_.AddVec(-1.0, ee_neg_, 1.0);
  ee_corr_.AddVec(lr, ee_pos_, momentum_);

  U_pos_.AddMat(-1.0, U_neg_, 1.0);
  U_corr_.AddMat(lr, U_pos_, momentum_);
  U_corr_.AddMat(wc, U_, 1.0);

  bt_.AddVec(1.0, bt_corr_, 1.0);
  lamt2_.AddVec(1.0, lamt2_corr_, 1.0);
  gamma2_.AddVec(1.0, gamma2_corr_, 1.0);
  d_.AddVec(1.0, d_corr_, 1.0);
  ee_.AddVec(1.0, ee_corr_, 1.0);
  U_.AddMat(1.0, U_corr_, 1.0);

  gamma2_.ApplyFloor(0.0);
  lamt2_.ApplyFloor(0.0);

}  // end Learn()

}  // end namespace

