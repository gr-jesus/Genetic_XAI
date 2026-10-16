#from Fourier_experiment import Experiment
import random
import numpy as np
import os
from tensorflow.keras.backend import clear_session
import subprocess
import time

class Chromosome():
	def __init__(self,
				 base_folder='../Kaggle_v3/',
		         r=1.0,
		         lr=0.00001,
		         beta_1=0.99,
		         beta_2=0.99,
		         epsilon=1e-7,
		         fitness=0.0,
		         accuracy=0.0,
		         exp_id=1):
		self.base_folder=base_folder	
		self.r=r 
		self.lr=lr
		self.beta_1=beta_1
		self.beta_2=beta_2
		self.epsilon=epsilon
		self.fitness=fitness
		self.accuracy=accuracy
		self.exp_id=exp_id
	


	def mutation(self):
		self.r=float(random.randrange(5,125,5))/100
		self.lr=float(random.randrange(1,70,1))/100000
		self.beta_1=float(random.randrange(5,99,1))/100
		self.beta_2=float(random.randrange(5,99,1))/100
		self.epsilon=float(random.randrange(1,100,1))/100000000

	def get_fitness(self):
		f = open('population.log','r')
		self.fitness=float(f.readlines()[-1].split(',')[-1].replace('\n', ''))
		f.close()

	def evaluate_chromosome(self):
		
		#os.system('python Fourier_experiment.py --dataset '+self.base_folder+' --lr '+str(self.lr)+' --beta_1 '+str(self.beta_1)+' --beta_2 '+str(self.beta_2)+' --eps '+str(self.epsilon)+' --r '+str(self.r)+' --exp_id '+str(self.exp_id))
		#' --eps '+str(self.epsilon)+' --r '+str(self.r)+' --exp_id '+str(self.exp_id))
		#print()
		#out_exp=subprocess.run(['python', 'Fourier_experiment.py', '--dataset', self.base_folder, '--lr', str(self.lr), '--beta_1', str(self.beta_1),
		#	                    '--beta_2', str(self.beta_2), '--eps', str(self.epsilon), '--r', str(self.r), '--exp_id', str(self.exp_id)],
		#			   capture_output=True,
		#			   text=True).stdout.split('\n','')[-2]
		process=subprocess.run(['python', 'Fourier_experiment.py', '--dataset', self.base_folder, '--lr', str(self.lr), '--beta_1', str(self.beta_1),
			                    '--beta_2', str(self.beta_2), '--eps', str(self.epsilon), '--r', str(self.r), '--exp_id', str(self.exp_id)],
					   capture_output=True, text=True)
		while(process.returncode != 0):
			time.sleep(5)
		#print(process.stdout.split('\n')[-2])
		#print(float(process.stdout.split('\n')[-2].replace(' ','').split(',')[-1]))
		self.fitness=float(process.stdout.split('\n')[-2].replace(' ','').split(',')[-1])
		self.accuracy=float(process.stdout.split('\n')[-2].replace(' ','').split(',')[-3])
		#exp=Experiment(base_folder=self.base_folder,
		#			   lr=self.lr,
		#			   beta_1=self.beta_1,
		#			   beta_2=self.beta_2,
		#			   epsilon=self.epsilon,
		#			   r=self.r)

		#self.fitness, self.accuracy=exp.run()
		# Delete the experiments when we obtain the fitness and accuracy
		#print(psutil.Process().memory_info().rss / (1024 * 1024))
		#del(exp)
		#print(psutil.Process().memory_info().rss / (1024 * 1024))

	def return_data(self):
		return [self.r, self.lr, self.beta_1, self.beta_2, self.epsilon]

	def print(self):
		print(self.r,',',self.lr,',', self.beta_1,',',self.beta_2,',',self.epsilon,',', self.fitness, self.accuracy)
		#print('r:',self.r)
		#print('lr:',self.lr)
		#print('beta_1:', self.beta_1)
		#print('beta_2:', self.beta_2)
		#print('epsilon:', self.epsilon)
		#print('fitness:', self.fitness)


#ch=Chromosome()
#ch.mutation()
#ch.evaluate_chromosome()
#ch.print()

class Genetic():
	def __init__(self,
				 intial_population_size=30,
				 mutation_rate=0.2,
				 crossover_rate=0.4,
				 crossover_mask=[0,0,1,1,1]):
		self.initial_population_size=intial_population_size
		self.crossover_rate=crossover_rate
		self.mutation_rate=mutation_rate
		self.population=[]
		self.crossover_mask=crossover_mask
		self.counter=0

	def create_random_chromosome(self):
		clear_session()
		ch=Chromosome(exp_id=self.counter)
		self.counter+=1
		ch.mutation()
		#ch.get_fitness()
		#ch.print()
		ch.evaluate_chromosome()
		return ch

	def initial_population(self):
		print('Initial population')
		for chrmsm in range(self.initial_population_size):
			self.population.append(self.create_random_chromosome())

	def mutate(self):
		
		selected=self.population[int(np.random.uniform(0,self.initial_population_size))].return_data()
		mutated=Chromosome(exp_id=self.counter)
		self.counter+=1
		mutated.mutation()
		mutated=mutated.return_data()
		p=int(np.random.uniform(0,5))
		#print(selected)
		selected[p]=mutated[p]
		mutated=Chromosome(r=selected[0], lr=selected[1], beta_1=selected[2], beta_2=selected[3], epsilon=selected[4])
		mutated.evaluate_chromosome()
		#mutated.print()
		return mutated
		

	def mutation(self):
		print('Mutation')
		# eliminate the percentage of the population
		
		for i in range(int(self.initial_population_size*self.mutation_rate)):
			self.population.append(self.mutate())
		self.sort_population()
		self.population=self.population[:self.initial_population_size]

	def sort_population(self):
		# get the fitness of each chromosome in the population
		fitness=[]
		for ch in self.population:
			fitness.append(ch.fitness)
		# get the positions of the best chromosomes
		indexes=np.argsort(np.array(fitness))[::-1]
		aux=[]
		for i in indexes:
			aux.append(self.population[i])
		self.population=aux
		del(aux)

	def crossover_two_chromosomes(self):
		# code r, lr, beta_1, beta_2, epsilon
		#print(len(self.population))
		rndm=-1.0
		while(rndm<0.0):
			rndm=int(np.random.normal(4,3))
		#print(rndm)
		ch_1=self.population[rndm].return_data()
		
		rndm=int(np.random.uniform(0,self.initial_population_size))
		#print(rndm)
		ch_2=self.population[rndm].return_data()
		
		#print(ch_1)
		#print(ch_2)

		new_ch=[]
		for i in range(len(self.crossover_mask)):
			if self.crossover_mask[i]==0:
				new_ch.append(ch_1[i])
			else:
				new_ch.append(ch_2[i])
		#print(new_ch)
		ch=Chromosome(r=new_ch[0], lr=new_ch[1], beta_1=new_ch[2], beta_2=new_ch[3], epsilon=new_ch[4], exp_id=self.counter)
		self.counter+=1
		ch.evaluate_chromosome()
		
		self.population.append(ch)


	def crossover(self):
		print('Crossover')
		for i in range(int(self.initial_population_size*self.crossover_rate)):
			self.crossover_two_chromosomes()
		self.sort_population()
		self.population=self.population[:self.initial_population_size]


	def print_population(self):
		print('r,lr,beta_1,beta_2,epsilon,fitness')
		for chrmsm in self.population:
			chrmsm.print()

	def print_log_line(self, line):
		f = open('population.log','a')
		f.write(line)
		f.close()

	def open(self, path):
		with open(path) as f:
			for i in f.readlines()[1:]:
				data_ch=i.replace('\n','').split(',')
				ch=Chromosome(r=float(data_ch[4]),
					          lr=float(data_ch[0]),
					          beta_1=float(data_ch[1]),
					          beta_2=float(data_ch[2]),
					          epsilon=float(data_ch[3]),
					          accuracy=float(data_ch[5]),
					          fitness=float(data_ch[6]))
				self.population.append(ch)

	def run(self):
		#self.print_log_line('Initial Population\n')
		#self.initial_population()
		self.open('population.log')
		self.sort_population()
		self.print_population()
		for i in range(10):
			#print(', Population '+str(i+1))
			self.sort_population()
			self.crossover()
			self.mutation()
			self.print_population()

#os.system('rm population.log')
os.system('touch population.log')
gen=Genetic()
gen.run()

#	def initial_population(self):




