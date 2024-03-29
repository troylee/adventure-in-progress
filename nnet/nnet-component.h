// nnet/nnet-component.h

// Copyright 2011  Karel Vesely

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.



#ifndef KALDI_NNET_COMPONENT_H
#define KALDI_NNET_COMPONENT_H


#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
// #include "nnet/nnet-nnet.h"

#include <iostream>

namespace kaldi {

// declare the nnet class so we can declare pointer
class Nnet;
    

/**
 * Abstract class, basic element of the network,
 * it is a box with defined inputs, outputs,
 * and tranformation functions interface.
 *
 * It is able to propagate and backpropagate
 * exact implementation is to be implemented in descendants.
 *
 * The data buffers are not included 
 * and will be managed from outside.
 */ 
class Component {

  // Polymorphic Component RTTI
 public: 
  /// Types of the net components
  typedef enum {
    kUnknown = 0x0,
     
    kUpdatableComponent = 0x0100, 
    kBiasedLinearity,
    kSharedLinearity, 
    kKrylovLinearity,
    kDropoutBL,
    kCMVNBL,
    kPosNegBL,
    kGaussBL,
    kMaskedBL,
    kLinBL,
    kHMMBL,
    kCodeBL,

    kActivationFunction = 0x0200, 
    kSoftmax, 
    kSigmoid,
    kRelu,
    kSoftRelu,

    kTranform =  0x0400,
    kRbm,
    kExpand,
    kCopy,
    kTranspose,
    kBlockLinearity,
    kBias,
    kWindow,
    kLog,
    kMaskedRbm,
    kRoRbm,
    kGRbm,
    kLinRbm
  } ComponentType;
  /// Pair of type and marker
  struct key_value {
    const Component::ComponentType key;
    const char *value;
  };
  /// Mapping of types and markers 
  static const struct key_value kMarkerMap[];
  /// Convert component type to marker
  static const char* TypeToMarker(ComponentType t);
  /// Convert marker to component type
  static ComponentType MarkerToType(const std::string &s);

  Component(MatrixIndexT input_dim, MatrixIndexT output_dim, Nnet *nnet) 
      : input_dim_(input_dim), output_dim_(output_dim), nnet_(nnet) { }
  virtual ~Component() { }
   
 public:
  /// Get Type Identification of the component
  virtual ComponentType GetType() const = 0;  
  /// Check if contains trainable parameters 
  virtual bool IsUpdatable() const { 
    return false; 
  }

  /// Get size of input vectors
  MatrixIndexT InputDim() const { 
    return input_dim_; 
  }  
  /// Get size of output vectors 
  MatrixIndexT OutputDim() const { 
    return output_dim_; 
  }
 
  /// Perform forward pass propagateion Input->Output
  void Propagate(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out); 
  /// Perform backward pass propagateion ErrorInput->ErrorOutput
  void Backpropagate(const CuMatrix<BaseFloat> &in_err,
                     CuMatrix<BaseFloat> *out_err); 

  /// Read component from stream
  static Component* Read(std::istream &is, bool binary, Nnet *nnet);
  /// Write component to stream
  void Write(std::ostream &os, bool binary) const;
  /// Write component to stream
  void WriteAsBiasedLinearity(std::ostream &os, bool binary) const;



  // abstract interface for propagation/backpropagation 
 protected:
  /// Forward pass transformation (to be implemented by descendents...)
  virtual void PropagateFnc(const CuMatrix<BaseFloat> &in,
                            CuMatrix<BaseFloat> *out) = 0;
  /// Backward pass transformation (to be implemented by descendents...)
  virtual void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err,
                                CuMatrix<BaseFloat> *out_err) = 0;

  /// Reads the component content
  virtual void ReadData(std::istream &is, bool binary) { }

  /// Writes the component content
  virtual void WriteData(std::ostream &os, bool binary) const { }


  // data members
 protected:
  MatrixIndexT input_dim_;  ///< Size of input vectors
  MatrixIndexT output_dim_; ///< Size of output vectors
  
  Nnet *nnet_; ///< Pointer to the whole network
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(Component);
};


/**
 * Class UpdatableComponent is a Component which has
 * trainable parameters and contains some global 
 * parameters for stochastic gradient descent
 * (learnrate, momenutm, L2, L1)
 */
class UpdatableComponent : public Component {
 public: 
  UpdatableComponent(MatrixIndexT input_dim, MatrixIndexT output_dim, Nnet *nnet)
    : Component(input_dim, output_dim, nnet),
      learn_rate_(0.0), momentum_(0.0), l2_penalty_(0.0), l1_penalty_(0.0), average_grad_(false) { }
  virtual ~UpdatableComponent() { }

  /// Check if contains trainable parameters 
  bool IsUpdatable() const { 
    return true; 
  }

  /// Compute gradient and update parameters
  virtual void Update(const CuMatrix<BaseFloat> &input,
                      const CuMatrix<BaseFloat> &err) = 0;

  /// Sets the learning rate of gradient descent
  void SetLearnRate(BaseFloat lrate) { 
    learn_rate_ = lrate; 
  }
  /// Gets the learning rate of gradient descent
  BaseFloat GetLearnRate() { 
    return learn_rate_; 
  }
  
  /// Sets momentum
  void SetMomentum(BaseFloat mmt) { 
    momentum_ = mmt; 
  }
  /// Gets momentum
  BaseFloat GetMomentum() { 
    return momentum_; 
  }

  /// Sets L2 penalty (weight decay)
  void SetL2Penalty(BaseFloat l2) { 
    l2_penalty_ = l2; 
  }
  /// Gets L2 penalty (weight decay)
  BaseFloat GetL2Penalty() { 
    return l2_penalty_; 
  }

  /// Sets L1 penalty (sparisity promotion)
  void SetL1Penalty(BaseFloat l1) { 
    l1_penalty_ = l1; 
  }
  /// Gets L1 penalty (sparisity promotion)
  BaseFloat GetL1Penalty() { 
    return l1_penalty_; 
  }

  /// Sets whether to use average gradients
  void SetAverageGrad(bool flag) {
    average_grad_ = flag;
  }
  /// Gets whether to use average gradients
  bool GetAverageGrad() {
    return average_grad_;
  }

 protected:
  BaseFloat learn_rate_; ///< learning rate (0.0..0.01)
  BaseFloat momentum_;   ///< momentum value (0.0..1.0)
  BaseFloat l2_penalty_; ///< L2 regularization constant (0.0..1e-4)
  BaseFloat l1_penalty_; ///< L1 regularization constant (0.0..1e-4)
  bool average_grad_;    ///< whether divide the gradient by number of samples
};




inline void Component::Propagate(const CuMatrix<BaseFloat> &in,
                                 CuMatrix<BaseFloat> *out) {
  if (input_dim_ != in.NumCols()) {
    KALDI_ERR << "Nonmatching dims, component:" << input_dim_ << " data:" << in.NumCols();
  }
  
  if (output_dim_ != out->NumCols() || in.NumRows() != out->NumRows()) {
    out->Resize(in.NumRows(), output_dim_);
  }

  PropagateFnc(in, out);
}


inline void Component::Backpropagate(const CuMatrix<BaseFloat> &in_err,
                                     CuMatrix<BaseFloat> *out_err) {
  if (output_dim_ != in_err.NumCols()) {
    KALDI_ERR << "Nonmatching dims, component:" << output_dim_ 
              << " data:" << in_err.NumCols();
  }
  
  if (input_dim_ != out_err->NumCols() || in_err.NumRows() != out_err->NumRows()) {
    out_err->Resize(in_err.NumRows(), input_dim_);
  }

  BackpropagateFnc(in_err, out_err);
}


} // namespace kaldi


#endif
