from numpy import exp, array, random, dot

class Neuron():
	def __init__(self,x,y):
		random.seed(1)
		self.synaptic_weights = 2 * random.random((x,y)) - 1
	
	def Process(self, inputs):
		return self.Sigmoid(dot(inputs, self.synaptic_weights))

	def Sigmoid(self,x):
		return 1 / (1 + exp(-x))

class NeuralNetwork():
	def __init__(self,inp,outp):
		self.training_inputs = inp
		self.training_outputs = outp
		self.neuron1 = Neuron(5,len(inp))
		self.neuron2 = Neuron(len(inp),1)
	

	
	def GradientSigmoid(self,x):
		return x * (1-x)
	
	def Learn(self,iterations):
		for iteration in range(iterations):
			output1 = self.neuron1.Process(training_inputs)

			output2 = self.neuron2.Process(output1)

			error2 = training_outputs - output2
			
			adjustment2 = error2 * self.GradientSigmoid(output2)
			error1 = dot(adjustment2,self.neuron2.synaptic_weights.T)
			
			adjustment1 = error1 * self.GradientSigmoid(output1)
			
			self.neuron2.synaptic_weights += dot(output1.T,adjustment2)
			self.neuron1.synaptic_weights += dot(training_inputs.T,adjustment1)
	
	def Process(self,inp):
		output1 = self.neuron1.Process(inp)
		return self.neuron2.Process(output1)
			

if __name__ == "__main__":
	
	training_inputs = array([[1,1,1,1,1], [1,0,0,0,1], [0,1,1,1,0], [0,1,0,0,1], [0,0,0,1,0], [1,1,0,0,0],[1,0,1,0,1], [0,0,1,1,0], [1,1,1,0,1],[1,0,1,1,0],[1,0,1,1,1]])
	training_outputs = array([[0,1,1,1,0,0,1,0,0,1,1]]).T
	neural_network = NeuralNetwork(training_inputs,training_outputs)
	neural_network.Learn(100000)
	
	print("New Synaptic Weights")
	print(neural_network.neuron1.synaptic_weights)
	print(neural_network.neuron2.synaptic_weights)
	
	print("What about inputs [0,0,1] and [1,0,1] and ")
	print(neural_network.Process(array([0,0,1,1,1])))
	print(neural_network.Process(array([1,0,1,0,0])))
	print(neural_network.Process(array([1,1,0,1,0])))
	print(neural_network.Process(array([0, 1, 0, 1, 0])))