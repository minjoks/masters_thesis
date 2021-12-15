

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from numpy import sqrt, diag, zeros, dot, abs

def homals(lG,ndim,niter,ldecomp, ndegree,test,train, cand, X_test, x_test, y_test):
  """
  # Sandor Szedmak
  
  Task: to compute the homals based Euclidean representation of multivariate
        variabels, where the variables can be categorical.
        See details:
  
          @article{Gifi1998,
          author={G. Michailidis and J. de Leeuw},
          title={The Gifi System of Descriptive Multivariate Analysis},
          journal={Statistical Science},
          volume={13, No. 4},
          pages={307-336},
          year={1998}
          }

  Input:      lG      list of 2d arrays, arrays belong to the views
                      each array has size:
                        [sample_size,variable dimension,
                                      e.g. number of category indicators, kernels]
              ndim    dimension of the representation space, the common factors 
              niter   number of iteration 
              ldecomp   list of indexes of weights need to be decomposed
                        see nonlinear principal component anlaysis
              xmissing  vector of indicators of missing examples
              ndegree  degree of the decomposition
  
  Output:     X       2d array sample item representation  sample item * ndim
              lY      list of 2d arrays [variable dimension of the view, ndim ]
                      weight component representation in theitem space

  """
  unb=cand
  xmissing=None
  nj=len(lG)          ## number of variables
  m=lG[0].shape[0]    ## number of sample items

  if ldecomp is None:
    ldecomp=[]
  if xmissing is None:
    xmissing=np.ones(m)
  
  lY=[]
  for j in range(nj):
    lY.append(np.zeros((lG[j].shape[1],ndim)))
  
  ## initialize X   
  X=np.random.randn(m,ndim)
  xmean=np.mean(X,axis=0)
  X=X-np.tile(xmean,(m,1))
  ## orthogonalization of columns
  U,S,V=np.linalg.svd(X,full_matrices=0)
  X=np.sqrt(ndim)*np.copy(U)

  ## compute invers covariance of G matrices, denoted as D in the paper
  lD=[]
  lDI=[]
  for j in range(nj):
    lD.append(np.dot(lG[j].T,lG[j]))
    lDI.append(np.linalg.pinv(lD[j]))
  
  err_prev=0
  for iiter in range(niter):
    ## compute Y's
    for j in range(nj):
      lY[j]=np.dot(np.dot(lDI[j],lG[j].T),X)
                 
    ## compute X
    X=np.zeros((m,ndim))
    for j in range(nj):
      X+=np.dot(lG[j],lY[j])
    X=X/nj
              
    xmean=np.mean(X,axis=0)
    X=X-np.tile(xmean,(m,1))
    ## orthogonalization of columns
    U,S,V=np.linalg.svd(X,full_matrices=0)
    X=np.sqrt(ndim)*np.copy(U)
    
    err_now=0
    for j in range(nj):
      err_now=np.sqrt(np.sum((X-np.dot(lG[j],lY[j]))**2))
    
    #print(iiter,err_now)

    #if abs(err_now-err_prev)<0.000001/nj:
     # break
    err_prev=err_now
    
    
  print("done one fold")

  #rename lY and X
  lW=lY
  Z=X
	
  z_tildes = np.zeros((len(test), ndim))
  xview = np.dot(cand[:,train], lW[2])
  sds = np.zeros((xview.shape[0], len(test)))
    
  for i in range(len(test)):
   z_tildes[i,:] = 1/2*(np.dot(lW[0].T, X_test[i,:]) + np.dot(lW[1].T, x_test[i,:]))
   xfactor = np.outer(np.ones(xview.shape[0]), z_tildes[i,:])
   sd = np.sum((xview-xfactor)**2,1)
   sds[:,i] = sd
    
  ranks = np.zeros((len(test),1))
   
  for j in range(len(test)):
      count = 0
      biggest = np.corrcoef(cand[np.argmin(sds[:,j]),train], y_test[j,:])[0,1]
      for k in range(unb.shape[0]):
    	  co = np.corrcoef(unb[np.argmin(sds[:,j])], cand[k,:])[0,1]
    	  if co>biggest:
    	      count = count + 1
			
      ranks[j] = count
	
  return(ranks)

# Define the MLP 
def mlp(n_inputs, n_outputs):
  model=Sequential()
  model.add(Dense(500, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  #model.add(Dense(1500, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  #model.add(Dense(400, activation="relu"))
  model.add(Dense(n_outputs))
  model.compile(loss='mean_squared_error', optimizer='adam')
  
  return model

def get_iokr_inverse(K, lmbda):

    n = K.shape[0]
    inverse = np.linalg.pinv(lmbda*np.eye(n)+K)

    return inverse


def get_iokr_preimage(k_preimg, K_y_preimg_set, inverse):

    [t, _] = K_y_preimg_set.shape
    end = np.dot(inverse, k_preimg)
    results = []
    for iii in range(t):
        results.append(np.dot(K_y_preimg_set[iii, :], end))
    return results
    
 
    
class cls_mmr_solver:

  def __init__(self):

    self.niter=500   ## maximum iteration
    self.normx1=1  ## normalization within the kernel by this power
    self.normy1=1  ## normalization within the kernel by this power
    self.normx2=1  ## normalization of duals bound
    self.normy2=1  ## normalization of duals bound
    ## self.ilabel=0  ## 1 explicit labels, 0 implicit labels
    ## self.ibias=0   ## 0 no bias considered, 1 bias is computed by solver
    ## self.ibias_estim=0  ## estimated bias =0 no =1 computed
    ## self.i_l2_l1=1   ## =1 l2 norm =0 l1 norm regularization   
    ## self.report=0

  ## ----------------------------------------------------
  def mmr_solver(self,Kx,Ky,C,D=0,qs0=None):
    """
    Sandor Szedmak
    Taks: to solve the unbiased mmr problem
    Input: Kx   2d square array input kernel
           Ky   2d square array output kernel
           C    penalty weight on the loss
           D    penalty weight on the negative loss   D <= \alpha <= C
           qs0  vector of scalling the margin, default is 1.
    """  
  ## solve an unbiased mmr
    maxiter=self.niter
    err_tolerance=0.001
    xeps=10**(-4)

  ## input output norms
    dx=diag(Kx)
    dy=diag(Ky)
    dx=dx+(abs(dx)+xeps)*(dx<=0)
    dy=dy+(abs(dy)+xeps)*(dy<=0)
    dKx=sqrt(dx)
    dKy=sqrt(dy)

    dKxy1=dKx**self.normx1*dKy**self.normy1   ## norm based scale of the margin
    dKxy2=dKx**self.normx2*dKy**self.normy2   ## norm based scale of the loss

    dKxy2+=1.0*(dKxy2==0)    ## to avoid zero

    lB=float(D)/(dKxy2)               ## scale the ranges
    uB=float(C)/(dKxy2)

    Bdiff=uB-lB
    h = Bdiff*Bdiff
    g = np.sum(h)

    z_eps=err_tolerance*sqrt(np.sum(Bdiff*Bdiff))

    if qs0 is None:
      qs=-dKxy1
    else:
      qs=-qs0*dKxy1

    Kxy=Kx*Ky
    m=Kxy.shape[0]
  ## scaling by diagonal elements  
    ## dKxy=diag(Kx)
    ## dKxy=dKxy+(dKxy==0)
    ## Kxy=Kxy/outer(dKxy,dKxy)
    ## qs=qs/dKxy

    for irow in range(m):
      if Kxy[irow,irow]==0:
        Kxy[irow,irow]=1

  ##  xalpha=zeros(m)
    xalpha=0.5*(uB+lB)
    xalpha0=xalpha.copy()
    for irepeat in range(maxiter):
      for irow in range(m):
        t=(-qs[irow]-dot(Kxy[irow],xalpha0))/Kxy[irow,irow]
        ## t=-qs[irow]-dot(Kxy[irow],xalpha0)
        xnew=xalpha0[irow]+t
        lbi=lB[irow]
        ubi=uB[irow]
        if lbi<xnew:
          if ubi>xnew:
            xalpha0[irow]=xnew
          else:
            xalpha0[irow]=ubi
        else:
          xalpha0[irow]=lbi
      xdiff=xalpha0-xalpha
      zerr=sqrt(np.sum(xdiff*xdiff))     ## L2 norm error
  ##     zerr=max(abs(xdiff))     ## L_infty norm error
      xalpha=xalpha0.copy()
      if zerr<z_eps:
  ##       print irepeat
        break
  ## xalpha the dual solution
    return(xalpha)    
    
    
    
    
    
    
    
    
    
 

