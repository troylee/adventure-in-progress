
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)

BINFILES = nnet-train-xent-hardlab-perutt \
        nnet-train-xent-hardlab-frmshuff \
	nnet-train-mse-tgtmat-frmshuff \
	nnet-train-mmi-sequential \
	nnet-train-mpe-sequential \
	rbm-train-cd1-frmshuff rbm-convert-to-nnet \
	nnet-forward nnet-copy nnet-info nnet-concat \
	transf-to-nnet cmvn-to-nnet copy-feats-from-text \
	copy-feats-to-text ideal-hidmask-forward \
	nnet-hidmask-mse-tgtmat-frmshuff \
	codeat-convert codeat-forward codeat-init-code \
	codeat-train codeat-train-parallel \
	codemat-init-code codemat-train codemat-forward \
	linat-forward

OBJFILES =



TESTFILES =

ADDLIBS = ../nnet/kaldi-nnet.a ../cudamatrix/kaldi-cudamatrix.a ../lat/kaldi-lat.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
					../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
