import os
import sys
from os.path import isfile, isdir, join
from pathlib import Path
from sklearn import mixture
import math
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial import distance
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import pickle
from hmmlearn import hmm
import time
import copy
# from helper_functions import genRTTM, removeSilence


def getMFCC(sig, rate, numcep = 19, nfilt = 26, winlen = 0.03, overlap = 0.01):
  # overlap = 0.01 #10 ms window shift
  # fullPath = join(path,fileName+'_processed.wav')
  mfcc_feat = mfcc(sig, rate, numcep = numcep, nfilt = nfilt, winlen=winlen, winstep=overlap)
  return mfcc_feat
  

def fitUnimodal(C):
  means = []
  covMatrices = []
  for c in C:
    means.append(np.mean(c, axis = 0))
    covMatrices.append(np.cov(c.T))
  return means, covMatrices

def calcYgivenX(x, GaussianMeans, GaussianCovMatrices, i):
  p = 0.0
  numOfClusters = len(GaussianMeans)
  w = 1.0/numOfClusters
  D = x.shape[0]
  probMat = np.zeros((D, num_of_clusters), dtype = float)
  for i in range(num_of_clusters):
    probMat[:,i] = multivariate_normal(x, GaussianMeans[i], GaussianCovMatrices[i])
  p = 0.0
  self.gamma = self.gamma/(np.sum(self.gamma, axis = 1)[:,None]) 
  return p

########################
##### IB Algorithm #####
########################


def init_prob(N, C, GaussianMeans, GaussianCovMatrices):
  probYgivenC = np.zeros((N,N), dtype = float)
  probCgivenX = np.zeros((N,N), dtype = float)
  print("IB probability Initialization Started.")
  for i in range(N):
    probCgivenX[i][i] = 1.0
  for i in range(N):
    temp1 = []
    temp2 = []
    x = C[i]
    w = 1.0/N
    D = x.shape[0]
    probMat = np.zeros((D, N), dtype = float)
    for j in range(N):
      probMat[:,j] = multivariate_normal.pdf(x, GaussianMeans[j], GaussianCovMatrices[j]).ravel()
    probMat = probMat/(np.sum(probMat, axis = 1)[:,None])
    probYgivenC[i] = np.mean(probMat, axis = 0)
    if i%50 == 0:
      print("Done: ",i)
  print("IB probability Initialization Completed.")
  print("#######################")
  return probYgivenC, probCgivenX

def init_delta(N, probX, probC, probYgivenC, probCgivenX, beta):
  print("Initializing Objective Function difference.")
  del_F = np.zeros((N, N), dtype = float)
  del_F[:,:] = np.inf
  probXgivenC = ((np.array(probCgivenX)*np.array(probX)).T/probC).T
  for i in range(N):
    for j in range(i+1, N): 
      temp1 = distance.jensenshannon(probYgivenC[:,i], probYgivenC[:,j]) 
      temp2 = distance.jensenshannon(probXgivenC[i], probXgivenC[j]) 
      dij = temp1 - (1/beta)*temp2
      del_F[i][j] = (probC[i] + probC[j])*dij
  print("Objective Function difference Initialization Completed.")
  print("#######################")
  return del_F

def aIB(N, num_of_speakers, ClusterMapping, probX, probC, probYgivenC, probCgivenX, del_F, beta):
  startTime = time.time()
  bestClusterMapping = copy.deepcopy(ClusterMapping)
  num_of_clusters = N

  while num_of_clusters > 1:
    #Find clusters with min difference in objective function
    mIdx = np.argmin(del_F)
    i = mIdx//N
    j = mIdx%N
    #Merge clusters
    probCr = probC[i] + probC[j]
    del_F[:,j] = np.inf
    del_F[j,:] = np.inf
    ClusterMapping[i] += ClusterMapping[j]
    ClusterMapping[j] = []
    probYgivenC[i] = (probYgivenC[i]*probC[i] + probYgivenC[j]*probC[j])/probCr
    probC[i] = probCr
    probCgivenX[i] = [0 for idx in probCgivenX[i]]

    #Reevaluating probabilities
    for idx in ClusterMapping[i]:
      probCgivenX[i][idx] = 1
    probXgivenC = ((np.array(probCgivenX)*np.array(probX)).T/probC).T
    for idx in range(0, i):
      if del_F[idx,i] == np.inf:
        continue
      temp1 = distance.jensenshannon(probYgivenC[:,idx], probYgivenC[:,i]) 
      temp2 = distance.jensenshannon(probXgivenC[idx], probXgivenC[i]) 
      dij = temp1 - (1/beta)*temp2
      del_F[idx][i] = (probC[idx] + probC[i])*dij
    for idx in range(i+1, N):
      if del_F[i, idx] == np.inf:
        continue 
      temp1 = distance.jensenshannon(probYgivenC[:,i], probYgivenC[:,idx]) 
      temp2 = distance.jensenshannon(probXgivenC[i], probXgivenC[idx]) 
      dij = temp1 - (1/beta)*temp2
      del_F[i][idx] = (probC[i] + probC[idx])*dij
    num_of_clusters = num_of_clusters-1
    
    if num_of_clusters == num_of_speakers:
      # print("Deep Copying dict:")
      bestClusterMapping = copy.deepcopy(ClusterMapping)

    if num_of_clusters%50 == 0 or num_of_clusters<=10:
      print("Clusters Remaining: ", num_of_clusters)
      # print("Time Elapsed: ", "{:.2f}".format((time.time()-startTime)/60), " minutes")
  # print("IB algo Completion Time: ", "{:.2f}".format((time.time()-startTime)/60), " minutes")
  print("IB Complete.")
  print("#######################")
  return bestClusterMapping
  # ClusterMapping = copy.deepcopy(bestClusterMapping)


def agglomerativeIB(mfcc_feat, N, num_of_speakers):
  ## Initialize Variables ##
  C = np.array_split(mfcc_feat, N)
  GaussianMeans, GaussianCovMatrices = fitUnimodal(C)
  ClusterMapping = dict(zip(range(N), [[i] for i in range(N)]))
  beta = 10.0
  probC = (1.0/N)*np.ones(N)
  probX = probC.copy()
  probYgivenC, probCgivenX = init_prob(N, C, GaussianMeans, GaussianCovMatrices)
  del_F = init_delta(N, probX, probC, probYgivenC, probCgivenX, beta)
  
  ## Run IB
  ClusterMapping = aIB(N, num_of_speakers, ClusterMapping, probX, probC, probYgivenC, probCgivenX, del_F, beta)
  return ClusterMapping


###################################
########## HMM Alignment ##########
###################################

def HMM_Align(N, mfcc_feat, ClusterMapping, init_cluster_len, num_of_speakers, hmm_gmm_cluster_num):
  print("HMM Alignment Begins")
  #Segregate MFCCs for each cluster for creating GMMs for HMM
  n, d = mfcc_feat.shape
  segCount = 0
  mfcc_segregated = []
  for key in ClusterMapping:
    if(len(ClusterMapping[key])>0):
      cur_mfcc = np.empty((0,d), dtype = float)
      for curSeg in sorted(ClusterMapping[key]):
        sIdx = max(0, (curSeg)*init_cluster_len)
        eIdx = min((curSeg+1)*init_cluster_len, n)
        cur_mfcc = np.concatenate((cur_mfcc, np.array(mfcc_feat[sIdx:eIdx])))
      mfcc_segregated.append(cur_mfcc)

  #Calculate parameters for initializing HMM
  gmmMeans = np.zeros((num_of_speakers, hmm_gmm_cluster_num,mfcc_feat.shape[-1]), dtype = float)
  gmmCov = np.zeros((num_of_speakers, hmm_gmm_cluster_num ,mfcc_feat.shape[-1], mfcc_feat.shape[-1]), dtype = float)
  gmmWeights = np.zeros((num_of_speakers, hmm_gmm_cluster_num ), dtype = float)
  for i, X in enumerate(mfcc_segregated):
    gmm = mixture.GaussianMixture(n_components=hmm_gmm_cluster_num).fit(X)
    gmmMeans[i] = gmm.means_
    gmmCov[i] = gmm.covariances_
    gmmWeights[i] = gmm.weights_

  #Separate the final clusters into a list
  cluster_list = []
  for i in range(N):
    if len(ClusterMapping[i]) !=0:
      cluster_list.append(ClusterMapping[i])

  #Map Which cluster belongs to which speaker
  cluster_to_speaker = np.ones((N,))
  for sp_id, clstr in enumerate(cluster_list):
    for c in clstr:
      cluster_to_speaker[c] = int(sp_id)

  #Start probability for HMM
  start_prob=np.zeros((num_of_speakers,))
  start_prob[int(cluster_to_speaker[0])] = 1.0

  #Calculate Transmission Probability for HMM
  transmission_prob = np.zeros((num_of_speakers,num_of_speakers), dtype = float)
  for i in range(1,N):
    fromSpeaker = int(cluster_to_speaker[i-1])
    toSpeaker = int(cluster_to_speaker[i])
    transmission_prob[fromSpeaker][toSpeaker] += 1
  for i in range(num_of_speakers):
    total = np.sum(cluster_to_speaker[:-1] == i)
    transmission_prob[i,:] /= total

  # Initializie HMM
  U_GMM_HMM =  hmm.GMMHMM(n_components = num_of_speakers,
                          n_mix = hmm_gmm_cluster_num,
                          covariance_type = "full",
                          init_params = "",
                          n_iter = 50
                          )
  U_GMM_HMM.covars_ = gmmCov
  U_GMM_HMM.means_ = gmmMeans
  U_GMM_HMM.weights_ = gmmWeights
  U_GMM_HMM.transmat_ = transmission_prob
  U_GMM_HMM.startprob_ = start_prob

  viterbiAligned = U_GMM_HMM.predict(mfcc_feat)
  print("HMM Alignment Completed")
  print("#######################")
  return viterbiAligned


#Generate RTTM file for predicted data
def createRTTM(n, viterbiAligned, init_cluster_time, fileName):
  #Generate final segments after viterbi alignment
  finalSegments = []
  path = "Data\\rttm\\"
  overlap = 0.01
  # init_cluster_timee = 2500 #2.5sec, is the min speaker segment size
  init_cluster_lenn = math.ceil(init_cluster_time/(overlap*1000))
  NN = math.ceil(n/init_cluster_lenn)
  fileName = fileName.split('.')[0]
  for i in range(NN):
    temp = viterbiAligned[i*init_cluster_lenn:min((i+1)*init_cluster_lenn, len(viterbiAligned))]
    x = np.bincount(temp).argmax()
    finalSegments.append(x)
  try:
    os.remove(join(path, fileName+'_pred.rttm'))
  except OSError:
    pass
  fileObj = open(join(path, fileName+'_pred.rttm'), 'a')
  idx = 0
  startTime = -1
  duration = 0
  # speakerId
  recName = fileName
  channelId = '1'
  onset = 0.000
  duration = 0.0
  while idx<=NN:
    if idx == NN or finalSegments[idx] != finalSegments[idx-1]:
      lineStr = 'SPEAKER ' + 'meeting' + ' ' + channelId + ' ' + str(onset) + ' ' + str(duration) + ' <NA> <NA> ' \
                    + 'speaker_'+str(int(finalSegments[idx-1])) + ' <NA> <NA>\n'
      if duration>0.0:
        fileObj.writelines(lineStr)
      onset += duration
      duration = init_cluster_time/1000
    else:
      duration += init_cluster_time/1000
    idx += 1
  fileObj.close()
  print("RTTM Generated successfully.")
  print("#######################")


#Generate RTTM file for predicted data
def createOtherRTTM(n, viterbiAligned, init_cluster_time, fileName):
  finalSegmentsMedFil = []
  path = "Data\\rttm\\"
  overlap = 0.01
  init_cluster_timee = 2500 #2.5sec, is the min speaker segment size
  init_cluster_lenn = math.ceil(init_cluster_timee/(overlap*1000))
  fileName = fileName.split('.')[0]
  for i in range(len(viterbiAligned)):
    sIdx = i
    eIdx = min(len(viterbiAligned), i+init_cluster_lenn)
    temp = viterbiAligned[sIdx:eIdx]
    x = np.bincount(temp).argmax()
    finalSegmentsMedFil.append(x)
  
  try:
    os.remove(join(path, fileName+'_predicted.rttm'))
  except OSError:
    pass
  fileObj = open(join(path, fileName+'_predicted.rttm'), 'a')
  idx = 0
  startTime = -1
  duration = 0
  channelId = '1'
  onset = 0.000
  duration = 0.0
  while idx<=len(viterbiAligned):
    if idx == len(viterbiAligned) or finalSegmentsMedFil[idx] != finalSegmentsMedFil[idx-1]:
      lineStr = 'SPEAKER ' + 'meeting' + ' ' + channelId + ' ' + str(onset) + ' ' + str(duration) + ' <NA> <NA> ' \
                    + 'speaker_'+str(int(finalSegmentsMedFil[idx-1])) + ' <NA> <NA>\n'
      if duration>0.0:
        fileObj.writelines(lineStr)
      onset += duration
      duration = 0.01
    else:
      duration += 0.01
    idx += 1
  fileObj.close()
  print("RTTM Generated successfully.")

def main(fileName, num_of_speakers):
  startTime = time.time()
  (rate,sig) = wav.read(join('Data\\audio\\', fileName))
  overlap = 0.01
  mfcc_feat = getMFCC(sig, rate)
  n, d = mfcc_feat.shape # n: total num of frames; d: num of features per frame
  init_cluster_time = 2500 #2.5sec
  init_cluster_len = math.ceil(init_cluster_time/(overlap*1000))
  N = math.ceil(n/init_cluster_len)
  num_of_clusters = N
  hmm_gmm_cluster_num = 5
  print("Num of frames: ", n)
  print("Num of initial segments: ", N)
  print("Num of features: ", d) 
  print("#######################")

  # IB Algorithm
  ClusterMapping = agglomerativeIB(mfcc_feat, N, num_of_speakers)

  #Viterbi Realignment
  viterbiAligned = HMM_Align(N, mfcc_feat, ClusterMapping, init_cluster_len, num_of_speakers, hmm_gmm_cluster_num)

  #Generate RTTM
  createRTTM(n, viterbiAligned, init_cluster_time, fileName)

  #Generate other RTTM
  createOtherRTTM(n, viterbiAligned, init_cluster_time, fileName)

  print("Done.")
  print("Total Time Elapsed: ", "{:.2f}".format((time.time()-startTime)/60), " minutes")

if __name__ == '__main__':
	argc = len(sys.argv)
	if argc == 1:
		main("ES2002a.wav",4)
	else:
		if argc == 2:
			main(sys.argv[1],4)
		else:
			main(sys.argv[1],int(sys.argv[2]))
