#encoding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os 

def read_aspfile(path):
	df = pd.read_csv(path, header=None, sep = '\s+', skiprows = 6).dropna()
	df.columns = ['index','x','y']
	data = np.array(df[['x','y']])
	return data

class Ant(object):
	def __init__(self, path = '../tsp/a280.tsp'):
		self.nodes = read_aspfile(path)
		self.length = self.nodes.shape[0]
		if self.length < 100:
			self.matrix = np.ones((self.length,self.length))
			self.dis_dict = self.init_disdict()
			self.etatable = self.init_etatable()
			self.ants = 40
			self.iterator = 200
			self.alpha = 1
			self.beta = 2
			self.decrese_ratio = 0.8
			self.Q = 1
			self.distance_avg = np.zeros((self.iterator))
			self.distance_best = np.zeros((self.iterator))
			self.path = np.zeros((self.iterator,self.length+1)).astype(int)

	def init_disdict(self):
		matrix = np.zeros((self.length,self.length))
		for i in range(self.length):
			for j in range(i,self.length):
				matrix[i,j] = matrix[j,i] = np.linalg.norm(self.nodes[i]-self.nodes[j])
		return matrix

	def init_etatable(self):
		return 1.0 / (self.dis_dict + np.diag([1e10]*self.length))

	def first_step(self):
		path_table = np.zeros((self.ants,self.length+1)).astype(int)
		if self.ants < self.length:
			path_table[:,0] = np.random.permutation(range(0,self.length))[:self.ants]
		else:
			path_table[:self.length,0] = np.random.permutation(range(0,self.length))[:]
			path_table[self.length:,0] = np.random.permutation(range(0,self.length))[:self.ants-self.length]
		return path_table

	def next_step(self, node, res_list):
		_list = res_list
		#set random list
		random_list = np.zeros(len(_list))
		#problist
		for i in range(len(res_list)):
			random_list[i] = np.power(self.matrix[node][_list[i]],self.alpha) * np.power(self.etatable[node][res_list[i]],self.beta)
		cumsumprobtrans = (random_list/np.sum(random_list)).cumsum()
		cumsumprobtrans -= np.random.rand()
		choice = _list[np.where(cumsumprobtrans>0)[0][0]]
		dis = self.dis_dict[node][choice]
		_list.remove(choice)
		return dis, choice, _list

	def refresh_matrix(self, path_table):
		refresh_table = np.zeros((self.length,self.length))
		for i in range(self.ants):
			for j in range(self.length):
				tmp = self.dis_dict[path_table[i,j]][path_table[i,j+1]]
				refresh_table[path_table[i,j]][path_table[i,j+1]] += ((self.Q / tmp) if tmp > 0 else 0)
		self.matrix = self.decrese_ratio *  self.matrix + refresh_table

	def refresh_statistic(self, iter, dis_table, path_table):
		self.distance_avg[iter] = dis_table.mean()
		if iter == 0:
			self.distance_best[iter] = dis_table.min()
			self.path[iter] = path_table[dis_table.argmin()].copy()
		else:
			if dis_table.min() > self.distance_best[iter-1]:
				self.distance_best[iter] = self.distance_best[iter-1]
				self.path[iter] = self.path[iter-1]
			else:
				self.distance_best[iter] = dis_table.min()
				self.path[iter] = path_table[dis_table.argmin()].copy()

	def one_iteration(self, iter):
		dis_table = np.zeros(self.ants)
		path_table = self.first_step()
		for num in range(self.ants):
			next_node = path_table[num,0]
			res_list = list(range(self.length))
			res_list.remove(next_node)
			for j in range(1,self.length):
				step_dis, next_node, res_list = self.next_step(next_node,res_list)
				dis_table[num] += step_dis
				path_table[num,j] = next_node
			dis_table[num] += self.dis_dict[path_table[num,0]][path_table[num,self.length-1]]
			path_table[num,self.length] = path_table[num,0]
		self.refresh_statistic(iter,dis_table,path_table)
		self.refresh_matrix(path_table)

	def start_cal(self):
		for index in range(self.iterator):
			self.one_iteration(index)
			if index % 20 == 0 and index > 0:
				print('iteration {} avg distance : {} best : {}'.format(index,self.distance_avg[index],self.distance_best[index]))
			
	def draw_plot(self, filepath):
		# 做出平均路径长度和最优路径长度        
		fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(12,10))
		axes[0].plot(self.distance_avg,'k',marker = u'')
		axes[0].set_title('Average Length')
		axes[0].set_xlabel(u'iteration')

		axes[1].plot(self.distance_best,'k',marker = u'')
		axes[1].set_title('Best Length')
		axes[1].set_xlabel(u'iteration')
		fig.savefig('../ans/{}_Average_Best.png'.format(filepath),dpi=500,bbox_inches='tight')
		plt.close()

		#作出找到的最优路径图
		bestpath = self.path[-1]

		plt.plot(self.nodes[:,0],self.nodes[:,1],'r.',marker=u'$\cdot$')
		plt.xlim([self.nodes[:,0].min(),self.nodes[:,0].max()])
		plt.ylim([self.nodes[:,1].min(),self.nodes[:,1].max()])

		for i in range(self.length):#
		    m,n = bestpath[i],bestpath[i+1]
		    plt.plot(np.array([self.nodes[m,0],self.nodes[n,0]]),np.array([self.nodes[m,1],self.nodes[n,1]]),'k')
		ax=plt.gca()
		ax.set_title("Best Path")
		ax.set_xlabel('X axis')
		ax.set_ylabel('Y_axis')
		plt.savefig('../ans/{}_Best_Path.png'.format(filepath),dpi=500,bbox_inches='tight')

def main():
	filepath_list = [i for i in os.listdir('../tsp/') if i[-4:] == '.tsp']
	for filepath in filepath_list:
		file_path = '../tsp/' + filepath
		print(file_path)
		try:
			ant = Ant(file_path)
			if ant.length < 100:
				ant.start_cal()
				ant.draw_plot(filepath)
		except:
			pass

if __name__ == '__main__':
	main()