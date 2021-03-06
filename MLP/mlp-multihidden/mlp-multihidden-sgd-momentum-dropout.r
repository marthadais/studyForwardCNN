#Copyright 2017 Martha Dais Ferreira, Rodrigo Fernandes de Mello
#
#This file is part of studyForwardCNN.
#
#studyForwardCNN is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#studyForwardCNN is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with ImageFalseNearest.  If not, see <http://www.gnu.org/licenses/>.
#
#See the file "COPYING" for the text of the license.
#
#Contact: 
#	Martha D. Ferreira: daismf@icmc.usp.br
#	Rodrigo Fernandes de Mello: mello@icmc.usp.br
#

require(nnet)

#################### BEGIN: ACTIVATION FUNCTIONS ######################

# ==========================
# Classical sigmoid function
# ==========================
#
# f(net) = 1 / (1 + exp(-net))
#
sigmoid <- function(net) {
	ret = 1.0 / (1.0 + exp(-net))
	return (ret)
}

#
# f'(net) = f(net) * (1 - f(net))
#
diff.sigmoid <- function(f, net) {
	if (!is.null(f)) return (f*(1-f))
	if (!is.null(net)) return (sigmoid(net)*(1-sigmoid(net)))
	return (NULL)
}

# =========================
# Classical linear function
# =========================
#
# f(net) = net
#
linear <- function(net) {
	return (net)
}

#
# f'(net) = 1
#
diff.linear <- function(f, net) {
	if (!is.null(f)) return (1)
	if (!is.null(net)) return (1)
	return (NULL)
}

#################### END: ACTIVATION FUNCTIONS ######################

mlp.architecture <- function(layers.sizes = c(1, 2, 1), 
			     f.net = sigmoid, df.dnet = diff.sigmoid) {

	if (length(layers.sizes) < 3) {
		return ("You must provide at least the input, one hidden and the output layers.")
	}

	layers = list()
	layers$hidden = list()

	for (layer.id in 1:(length(layers.sizes)-2)) {

		layers$hidden[[layer.id]] = matrix(runif(min=-1, max=1, 
			n=layers.sizes[layer.id+1]*(layers.sizes[layer.id]+1)), 
			nrow=layers.sizes[layer.id+1], 
			ncol=layers.sizes[layer.id]+1)
		#		w1	w2	theta
		# neuron 0	0.01   -0.9     -0.7
		# neuron 1	0.78	0.34	-0.3
		#
	}

	layer.id = length(layers.sizes)
	layers$output = matrix(runif(min=-1, max=1, 
			n=layers.sizes[layer.id]*(layers.sizes[layer.id-1]+1)), 
			nrow=layers.sizes[layer.id], 
			ncol=layers.sizes[layer.id-1]+1)
	#		w1	w2	theta
	# neuron 0	0.01   -0.9     -0.7
	#

	ret = list()
	ret$layers.sizes = layers.sizes
	ret$layers = layers
	ret$f.net = f.net
	ret$df.dnet = df.dnet

	return (ret)
}

# Apenas produzir a saida
forward <- function(model, x, dropout.probability = 0) {

	# Aplicando nas camadas escondidas
	input = list()
	net_h = list()
	f_net_h = list()

	input[[1]] = c(as.vector(ts(x)), 1)
	for (layer.id in 1:(length(model$layers.sizes)-2)) {
		net_h[[layer.id]] = 
			model$layers$hidden[[layer.id]] %*% input[[layer.id]]
		f_net_h[[layer.id]] = model$f.net(net_h[[layer.id]])

		# Begin: Dropping out...
		rnd = runif(min=0, max=1, n=length(f_net_h[[layer.id]]))
		dropout.ids = which(rnd < dropout.probability)
		net_h[[layer.id]][dropout.ids] = 0
		f_net_h[[layer.id]][dropout.ids] = 0
		# End: Dropping out...

		input[[layer.id+1]] = c(as.vector(ts(f_net_h[[layer.id]])), 1)
	}

	# Aplicando na camada de saída
	last = length(input)
	net_o = model$layers$output %*% input[[last]]
	f_net_o = model$f.net(net_o)

	ret = list()
	ret$input = input
	ret$f_net_h = f_net_h
	ret$f_net_o = f_net_o

	return (ret)
}

# Alterar pesos w e theta
backpropagation.sgd <- function(model, dataset, batch.size=1, eta=0.1, 
				threshold=1e-3, k=10, mult.factor=0.99,
				error_threshold = 1e-7, 
				momentum.alpha = 0.9,
				dropout.probability = 0.5) {

	input.size = model$layers.sizes[1]
	class.columns = seq(input.size + 1, ncol(dataset))

	# Corte no conjunto de dados
	X = matrix(dataset[,1:input.size], ncol=input.size)
	Y = matrix(dataset[,class.columns], nrow=nrow(X))

	if (batch.size > nrow(X)) {
		return ("batch.size must be smaller than the number of examples.")
	}

	nbatches = floor(nrow(X) / batch.size)

	cat("Enter to start running...")
	readline()

	squared_error_list = c()
	squared_error = threshold * 2
	old_squared_error = squared_error * 2

	counter = 0

	while (squared_error > threshold && 
	       		abs(old_squared_error - squared_error) > error_threshold) {

		old_squared_error = squared_error
		squared_error = 0

		# Randomizing the examples to ensure fairness
		example.ids = sample(1:nrow(X))
		X = X[example.ids,]
		Y = Y[example.ids,]

		deltaW_Output = 0
		deltaW_Hidden = list()

		for (nid in 1:(length(model$layers.sizes)-2)) {
		   deltaW_Hidden[[nid]] = 0
		}

		# For each batch
		for (batch.id in 1:nbatches) {

		   start = (batch.id-1)*batch.size+1
		   end = start + batch.size - 1
		   if (batch.id == nbatches) end = nrow(X)

		   nabla_Output = 0
		   nabla_Hidden = list()

		   for (nid in 1:(length(model$layers.sizes)-2)) {
			nabla_Hidden[[nid]] = 0
		   }

		   # For the examples in such batch
		   for (p in start:end) {
			# aplicar na mlp a entrada
			fwd = forward(model, X[p,], dropout.probability)
		        last = length(fwd$input)

			# Calculando o erro
			error = (Y[p,] - fwd$f_net_o)

			# Squared error para determinar critério de parada
			squared_error = squared_error + sum(error^2)

			# delta de output
			delta_o = error * model$df.dnet(f=fwd$f_net_o)

			# Computing all deltas for the hidden layers
			ncol = ncol(model$layers$output)
			delta = delta_o
			W = model$layers$output[,1:(ncol-1)]
			delta_h = list()
			for (layer.id in (length(model$layers.sizes)-2):1) {
				# Hidden Layer n
				delta_h[[layer.id]] = 
					model$df.dnet(f=fwd$f_net_h[[layer.id]]) * 
						sum(as.vector(delta)%*%W)
				# Atualizando variáveis
				ncol = ncol(model$layers$hidden[[layer.id]])
				W = model$layers$hidden[[layer.id]][,1:(ncol-1)]
				delta = delta_h[[layer.id]]
			}

			nabla_Output = nabla_Output + delta_o %*% fwd$input[[last]]
		   	for (layer.id in (length(model$layers.sizes)-2):1) {
			   
			   #if () {
			   #   nabla_Hidden[[layer.id]] = 
			   #	      delta_h[[layer.id]] %*% fwd$input[[layer.id]]
			   #} else {
			      nabla_Hidden[[layer.id]] = nabla_Hidden[[layer.id]] + 
				   delta_h[[layer.id]] %*% fwd$input[[layer.id]]
			   #}
			}
		   }

		   # Aprendizado

		   deltaW_Output = eta * (nabla_Output / (end-start+1)) + momentum.alpha * deltaW_Output
		   for (layer.id in (length(model$layers.sizes)-2):1) {
			deltaW_Hidden[[layer.id]] = 
				eta * (nabla_Hidden[[layer.id]] / (end-start+1)) +
					    	momentum.alpha * deltaW_Hidden[[layer.id]]
		   }

		   model$layers$output = model$layers$output + deltaW_Output

		   for (layer.id in (length(model$layers.sizes)-2):1) {
				model$layers$hidden[[layer.id]] =
					model$layers$hidden[[layer.id]] + deltaW_Hidden[[layer.id]]
		   }
		}

		squared_error = squared_error / nrow(X)

		squared_error_list = c(squared_error_list, squared_error)

		if (length(squared_error_list) >= k) {
			ntotal = length(squared_error_list)
			seq = as.numeric((ntotal-k+1):ntotal)
			values = squared_error_list[seq]

			scale = seq(0, 1, length=k)
			angle = lm(scale ~ values)$coefficients[2]
			if (is.na(angle)) { angle = 0 }

			if (angle > 0) {
				eta = eta * mult.factor
			} 
		}

		counter = counter + 1
		if (counter %% 100 == 0) {
			cat("Squared error = ", squared_error, " eta = ", eta, "\n")
		}
	}
	
	return (model)
}

mlp.test <- function(model, dataset, debug=T) {

	input.size = model$layers.sizes[1]
	class.columns = seq(input.size + 1, ncol(dataset))

	# Corte no conjunto de dados
	X = matrix(dataset[,1:input.size], ncol=input.size)
	Y = matrix(dataset[,class.columns], nrow=nrow(X))

	cat("Enter to start testing...")
	readline()

	output = NULL

	for (p in 1:nrow(X)) {
		# aplicar na mlp a entrada
		fwd = forward(model, X[p,], 0)

		if (debug) {
			cat("Input pattern = ", as.vector(X[p,]), 
			    " Expected output = ", as.vector(Y[p,]), 
				" Obtained output = ", as.vector(fwd$f_net_o), "\n")
		}

		output = rbind(output, as.vector(fwd$f_net_o))
	}

	return (output)
}

# x = iris.test(layers.sizes=c(4, 10, 5, 3), batch.size=75, eta=1, momentum.alpha=0.99, threshold=1e-7)
iris.test <- function(layers.sizes = c(4, 3, 3), batch.size=25,
		      	eta=0.5, threshold=1e-7, debug=F,
			error_threshold = 1e-7, momentum.alpha = 0.9,
			dropout.probability = 0.5, mult.factor = 0.99) {

	dataset = as.matrix(cbind(iris[,1:4], class.ind(iris[,5])))

	for (i in 1:4) {
		dataset[,i] = (dataset[,i] - mean(dataset[,i])) / sd(dataset[,i])
	}

	model = mlp.architecture(layers.sizes = layers.sizes, f.net=sigmoid, df.dnet=diff.sigmoid)
	trained.model = backpropagation.sgd(model, dataset, 
				batch.size=batch.size,
				eta=eta, threshold=threshold,
				error_threshold = error_threshold,
				momentum.alpha=momentum.alpha,
				dropout.probability=dropout.probability,
				mult.factor = mult.factor)
	return (mlp.test(trained.model, dataset, debug=debug))
}

