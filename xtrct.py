import xml.etree.ElementTree as ET
import numpy as np
import pickle
from itertools import permutations
ns = {'nite':"http://nite.sourceforge.net/", 'id':"EN2001a.A.segs"}
alls = ['A','B','C','D','E','F','G','H','I','J','K',"L"]
Vec = np.zeros((3000005,))
vuv_frames = np.load("vuv.npy")
Res = None
Seg_Sp = []
def main():
	d=0
	# for i in alls[:4]:
	# 	parse(i,d+1)
	# 	d+=1
	# np.save('Vec.npy',Vec)
	# V_Sp()
	Clstr()


def parse(S,i):	
	tree = ET.parse('./Data/EN2002b.'+S+'.segments.xml')

	root = tree.getroot()
	for seg in root.findall('segment'):
		st = seg.get('transcriber_start')
		end = seg.get('transcriber_end')
		st = int(float(st)*1000/10)
		end = int(float(end)*1000/10)
		for k in range(st,end):
			Vec[k]=i
def V_Sp():
	global Res,Seg_Sp
	Res = []
	Seg_Sp=[]
	for i in range(vuv_frames.shape[0]):
		if(vuv_frames[i]==1):
			for j in range(3):
				Res.append( Vec[i*3+j] )
	Res = np.array(Res).astype(np.int)
	print(np.sum(Res),np.sum(Vec),"----------")
	np.save("VAD_seg.npy",Res)
	for i in range(812):
		MxSp = np.zeros((5,))
		for j in range(250):
			try:
				MxSp[Res[i*250+j]]+=1
			except:
				pass
		if(MxSp[np.argmax(MxSp[1:]) +1]== 0):
			print(MxSp)
			Seg_Sp.append(0)
		else:
		# if 1:
			Seg_Sp.append(np.argmax(MxSp[1:])+1)
	np.save('Seg_Sp',Seg_Sp)
	print(Seg_Sp)
def Clstr():
	cid = np.load("clstr.npy").astype(np.int)
	aid = np.load("Seg_Sp.npy").astype(np.int)

	print(cid.shape,"\n\n",aid.shape)
	P = set(permutations([1,2,3,4]))
	Tscr = 0
	for i in P:
		scr=0
		T =0
		for j in range(811):
			if aid[j]:
				if aid[j] == i[cid[j]-1]:
					scr+=1
				T+=1

		if(scr>Tscr):
			Tscr =scr

	print(Tscr*1.0/T+ (812-T)/812)

	return

if __name__ == '__main__':
	main()
