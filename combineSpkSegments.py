import xml.etree.ElementTree as ET
import numpy as np
import pickle
from itertools import permutations
from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io
import sys
ns = {'nite':"http://nite.sourceforge.net/", 'id':"EN2001a.A.segs"}
alls = ['A','B','C','D','E','F','G','H','I','J','K',"L"]
All_t = []
Act_t = []
Mod_t = []
def parse(i,fn):
	global All_t
	S = alls[i]	
	tree = ET.parse('../ami_annotation/segments/EN2002b.'+S+'.segments.xml')

	root = tree.getroot()
	for seg in root.findall('segment'):
		st = seg.get('transcriber_start')
		end = seg.get('transcriber_end')
		st = float(st)
		end = float(end)
		All_t.append([st,end,i])
def VAD(fn):
	global Act_t,Mod_t
	samplerate, data = wavfile.read(fn+".wav")
	prev = int(0)
	sound = []
	for a in Act_t:
		st = int(a[0]*samplerate)
		ed = int(a[1]*samplerate)
		diff = (ed-st)
		sound.extend( data[st:ed])
		Mod_t.append([prev,prev + diff, a[2]])
		prev = prev + diff
		# print([prev/samplerate,(prev + diff)/samplerate, a[2]])

	wavfile.write(fn+"_vad.wav", samplerate, np.array(sound) )
	with open(fn+'_Mod.rttm','w') as fw:
		for i in Mod_t:
			fw.write("SPEAKER meeting 1 {0:0.3f} {1:0.3f} <NA> <NA> speaker_{2:d} <NA> <NA>\n".format(i[0]/samplerate,(i[1]-i[0])/samplerate,i[2]))
	np.save(fn+'_Mod_Seg.npy',Mod_t)
	print(fn+"_vad.wav")
def main(fn,ct):
	global All_t,Act_t,Mod_t
	for i in range(ct):
		parse(i,fn)
	All_t.sort()
	ln = len(All_t)
	# temp = []
	for i in range(1,ln):
		Act_t.append([All_t[i-1][0],min(All_t[i][0],All_t[i-1][1] ) ,All_t[i-1][2] ])
		# print([All_t[i-1][0],min(All_t[i][0],All_t[i-1][1] ) ,All_t[i-1][2] ],end=" ")
		# if(All_t[i][0] < All_t[i-1][1]):
			# print(" ** ",end=" ")
		# print(All_t[i][0],All_t[i-1][1])
	Act_t.append([All_t[ln-1][0],All_t[ln-1][1] ,All_t[ln-1][2] ])
	
	VAD(pjoin("..","Data",fn,"audio",fn+".Mix-Headset"))
	VAD(pjoin("..","Data",fn,"audio",fn+".Mix-Lapel"))

	# print(Act_t)

if __name__ == '__main__':
	argc = len(sys.argv)
	if argc == 1:
		main("ES2002a",4)
	else:
		if argc == 2:
			main(sys.argv[1],4)
		else:
			main(sys.argv[1],int(sys.argv[2]))