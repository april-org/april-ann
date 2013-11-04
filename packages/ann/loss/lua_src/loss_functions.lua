 ---------------------------
-- BINDING DOCUMENTATION --
---------------------------

april_set_doc("ann.loss",
	      {
		class="namespace",
		summary="Namespace which contains all loss functions",
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss",
	      {
		class="class",
		summary="Abstract loss function class, implements methods",
		description={
		  "The loss functions are child classes of this parent abstract",
		  "class.",
		  "The parent class implements the common API.",
		  "A loss function computes the error produce between a",
		  "token (from the output of an ann.component) and a target",
		  "token (the ground truth). The loss is accumulated in order",
		  "to compute the mean loss in a dataset of patterns.",
		  "The accumulated error is set to zero with reset method.",
		},
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.accum_loss",
	      {
		class="method",
		summary="Receives a loss matrix and accumulates it",
		params={
		  "The loss matrix computed by compute_loss method.",
		},
		outputs = {
		  "The given loss matrix",
		}
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.compute_loss",
	      {
		class="method",
		summary="Computes the loss between two tokens (input and target), but doesn't accumulate it",
		description={
		  "The loss is computed for a given input and target tokens.",
		  "This method returns a matrix with the loss for every given pair of patterns,",
		  "but it is not accumulated. Call to accum_loss to accumulate it to the internal state.",
		},
		params={
		  "Input token",
		  "Target token",
		},
		outputs = {
		  "A matrix with the loss computed for every pair of patterns (tokens)",
		},
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.gradient",
	      {
		class="method",
		summary="Computes the gradient of the loss between two tokens",
		description={
		  "The gradient is computed for a given input and target tokens.",
		  "This method returns the gradient for the given pair of patterns",
		  "(or for the",
		  "given bunch if more than one pattern is represented at the",
		  "tokens).",
		},
		params={
		  "Input token",
		  "Target token",
		},
		outputs = {
		  "The gradient computed for this pair of tokens",
		},
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.get_accum_loss",
	      {
		class="method",
		summary="Returns the mean loss from the last reset call",
		outputs = {
		  "The mean loss",
		  "The sample variance of the loss",
		},
	      })


-------------------------------------------------------------------

april_set_doc("ann.loss.reset",
	      {
		class="method",
		summary="Sets to zero the accumulated loss",
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.clone",
	      {
		class="method",
		summary="Deep copy of the object",
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.mse",
	      {
		class="class",
		summary="Mean square error loss function",
		description={
		  "The mse loss function computes 1/(2N)*\\sum_i\\sum_j (o^(i)_j - t^(i)_j)^2.",
		}
	      })

april_set_doc("ann.loss.mse.__call",
	      {
		class="method",
		summary="Constructor",
		params={
		  "The expected pattern size, 0 for a dynamic size layer"
		},
		outputs={ "An instance of ann.loss.mse" },
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.mae",
	      {
		class="class",
		summary="Mean absolute error loss function",
		description={
		  "The mae loss function computes 1/N*\\sum_i 1/M \\sum_j |o^(i)_j - t^(i)_j|.",
		}
	      })

april_set_doc("ann.loss.mae.__call",
	      {
		class="method",
		summary="Constructor",
		params={
		  "The expected pattern size, 0 for a dynamic size layer"
		},
		outputs={ "An instance of ann.loss.mae" },
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.cross_entropy",
	      {
		class="class",
		summary="Cross-entropy loss function (for two-class problems)",
		description={
		  "The cross-entropy loss function computes",
		  "1/N*\\sum_i [ t^(i) log o^(i) + (1-t^(i)) log (1-o^(i)).",
		  "It only works with log_logistic activation funtions.",
		}
	      })

april_set_doc("ann.loss.cross_entropy.__call",
	      {
		class="method",
		summary="Constructor",
		params={
		  "The expected pattern size, 0 for a dynamic size layer"
		},
		outputs={ "An instance of ann.loss.cross_entropy" },
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.multi_class_cross_entropy",
	      {
		class="class",
		summary="Multi-class cross-entropy loss function",
		description={
		  "The cross-entropy loss function computes",
		  "1/N*\\sum_i\\sum_j t^(i)_j log o^(i)_j.",
		  "It only works with log_logistic or log_softmax",
		  "activation funtions,",
		  "and is mandataory to have more than two output units.",
		}
	      })

april_set_doc("ann.loss.multi_class_cross_entropy.__call",
	      {
		class="method",
		summary="Constructor",
		params={
		  "The expected pattern size, 0 for a dynamic size layer"
		},
		outputs={ "An instance of ann.loss.multi_class_cross_entropy" },
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.batch_fmeasure_micro_avg",
	      {
		class="class",
		summary="The FMeasure computed from a bunch of patterns",
		description={
		  "This loss function computes",
		  "FMeasure for a given bunch of patterns, and for",
		  "multi-class models, using micro-averaging strategy.",
		}
	      })

april_set_doc("ann.loss.batch_fmeasure.__call",
	      {
		class="method",
		summary="Constructor",
		params={
		  ["size"]="The expected pattern size",
		  ["beta"]={
		    "The beta parameter of FMeasure",
		    "[optional], by default is 1.0",
		  },
		  ["complement"]={
		    "Boolean value, if true computes 1-output",
		    "[optional], by default is false",
		  },
		},
		outputs={ "An instance of ann.loss.batch_fmeasure" },
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.batch_fmeasure_macro_avg",
	      {
		class="class",
		summary="The FMeasure computed from a bunch of patterns",
		description={
		  "This loss function computes",
		  "FMeasure for a given bunch of patterns, and for",
		  "multi-class models, using macro-averaging strategy.",
		}
	      })

april_set_doc("ann.loss.batch_fmeasure.__call",
	      {
		class="method",
		summary="Constructor",
		params={
		  ["size"]="The expected pattern size",
		  ["beta"]={
		    "The beta parameter of FMeasure",
		    "[optional], by default is 1.0",
		  },
		  ["complement"]={
		    "Boolean value, if true computes 1-output",
		    "[optional], by default is false",
		  },
		},
		outputs={ "An instance of ann.loss.batch_fmeasure" },
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.zero_one",
	      {
		class="class",
		summary="The 0-1 loss function",
		description={
		  "The 0-1 loss function. This loss function is",
		  "derivable and therefore the gradient couldn't be computed.",
		},
	      })

april_set_doc("ann.loss.zero_one.__call",
	      {
		class="method",
		summary="Constructor",
		params={
		  "The expected pattern size",
		  {
		    "Threshold to consider activated the output neuron",
		    "when size=1 [optional]. By default is 0.5.",
		  },
		},
		outputs={ "An instance of ann.loss.zero_one" },
	      })

