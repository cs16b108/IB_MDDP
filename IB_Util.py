import os
from os.path import isfile, isdir, join
from pathlib import Path
from sklearn import mixture
import math
import numpy as np
import random
from scipy.spatial import distance
import scipy.io.wavfile as wav
import pickle
import time
import copy
import xml.etree.ElementTree as ET
from pydub import AudioSegment
import sys

fileName = 'ES2016b'
alls = ['A','B','C','D','E','F','G','H','I','J','K',"L"]

################################################
#### Generate Segment file (.scp) for IDIAP ####
################################################
def generateSCP(vuv_frames):
    fobj = open(fileName+'.scp', 'a')
    L = len(vuv_frames)
    i = 0
    while i<L:
        if vuv_frames[i] == 1:
            j = i
            cnt = 1
            while j+1<L and vuv_frames[j+1] == 1:
                cnt += 1
                j += 1
            curLine = fileName+'_'+str(i*3)+'_'+str(j*3)+'='+fileName+'.fea['+str(i*3)+','+str(j*3)+']\n'
            fobj.writelines(curLine)
            i = j
        i += 1
    fobj.close()

###########################################################
#### Remove Silence from audio file using ground truth ####
###########################################################
def removeSilence(fileName):
    audio = AudioSegment.from_wav(join('Data\\amicorpus\\',fileName,'audio', fileName+'.Mix-Headset.wav'))
    duration = len(audio) #In milliseconds
    newAudio = AudioSegment.empty()
    fObj = open(join('temp_files\\',fileName+'_with_silence.rttm'), 'r')
    lst = fObj.readlines()
    curStIdx = 0
    curEnIdx = 0
    lastStIdx = -1
    lastEnIdx = -1
    for i in range(len(lst)+1):
        if i != len(lst):
            l = lst[i]
            curStIdx = int(float(l.split()[3])*1000)
            curEnIdx = min(int((float(l.split()[3]) + float(l.split()[4]))*1000), len(audio))
        if i == 0:
            lastStIdx = curStIdx
            lastEnIdx = curEnIdx
        elif i == len(lst) or curStIdx>lastEnIdx:
            newAudio += audio[lastStIdx:lastEnIdx+1]
            lastStIdx = curStIdx
            lastEnIdx = curEnIdx
        else:
            lastEnIdx = max(curEnIdx, lastEnIdx)
    newAudio.export(join('Data\\audio\\', fileName+'.wav'), format="wav")
    fObj.close()

###############################################
#### Generate rttm from amicorpus segments ####
###############################################
def parse(S, diarizeReference, speaker_num, fileName):
	tree = ET.parse('ami_public_manual_1.6.2\\segments\\'+fileName+'.'+S+'.segments.xml')
	root = tree.getroot()
	for seg in root.findall('segment'):
		st = seg.get('transcriber_start')
		end = seg.get('transcriber_end')
		diarizeReference.append([float(st), float(end), 'speaker_'+str(int(speaker_num))])
	return diarizeReference

def rttmUtil(fileName, numOfSpeakers):
    diarizeReference = []
    for i in range(numOfSpeakers):
        diarizeReference = parse(alls[i],diarizeReference, i, fileName)
    diarizeReference.sort()
    # return diarizeReference
    try:
        os.remove(join('temp_files\\', fileName+'_with_silence.rttm'))
    except:
        pass
    fileObj = open(join('temp_files\\', fileName+'_with_silence.rttm'), 'a')
    rttm_out = []
    for d in diarizeReference:
        lineStr = 'SPEAKER ' + 'meeting' + ' ' + '1' + ' ' + str(d[0]) + ' ' + str(d[1]-d[0]) + ' <NA> <NA> ' + d[2] + ' <NA> <NA>\n'
        rttm_out.append(lineStr)
        fileObj.writelines(lineStr)
    fileObj.close()
    # os.remove()
    # return rttm_out

#Remove Silence from RTTM
def genRTTM(fileName, numOfSpeakers):
    rttmUtil(fileName, numOfSpeakers)
    try:
        os.remove(join('Data\\rttm\\',fileName+'_truth.rttm'))
    except:
        pass
    fObjRead = open(join('temp_files\\',fileName+'_with_silence.rttm'),'r')
    fObjWrite = open(join('Data\\rttm\\',fileName+'_truth.rttm'),'a')
    lastEnd = 0.0
    lines = fObjRead.readlines()
    i = 0
    L = len(lines)
    silences = 0.0
    newOutput = ''
    while i<L:
        s = lines[i].split()
        curStart = float(s[3])
        if curStart>=lastEnd:
            silences += curStart-lastEnd
        lineStr = 'SPEAKER ' + 'meeting' + ' ' + '1' + ' ' + str(float(curStart) - silences) + ' ' + s[4] + ' <NA> <NA> ' \
                            + s[7] + ' <NA> <NA>\n'
        # lst[3] = str(lastEnd)
        lastEnd = max(lastEnd, curStart + float(s[4]))
        fObjWrite.writelines(lineStr)
        i += 1
    fObjRead.close()
    fObjWrite.close()

def main(fileName, numOfSpeakers):
    # filePath = join('Data\\amicorpus',fileName, 'audio', fileName+'.Mix-Headset.wav')
    genRTTM(fileName, numOfSpeakers)
    print("RTTM Ground Truth generated.")
    time.sleep(2)
    removeSilence(fileName)
    print("Audio file without silence created.")


if __name__ == '__main__':
	argc = len(sys.argv)
	if argc == 1:
		main("ES2002a",4)
	else:
		if argc == 2:
			main(sys.argv[1],4)
		else:
			main(sys.argv[1],int(sys.argv[2]))

# ########################################
# #### Generate SCP from PRAAT Output ####
# ########################################
# f = open('IS1000a_Mix-Headset_25', 'r') #Give praat output file name
# fw = open('IS1000a.scp', 'a') #Give scp file name to be generated
# lst = f.readlines()
# # scp_vec = np.zeros((227976), dtype=int)
# for i in range(13, len(lst)-2):
#   if lst[i+2] == '"sounding"\n':
#     l = int(float(lst[i])*100)
#     r = int(float(lst[i+1])*100)
#     curLine = 'IS1000a_'+str(l)+'_'+str(r)+'=IS1000a.fea['+str(l)+','+str(r)+']\n' ##Change according to the name of scp file
#     fw.writelines(curLine)     
# f.close()
# fw.close()





