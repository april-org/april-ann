---------------------------
-- BINDING DOCUMENTATION --
---------------------------

april_set_doc("ann.loss",
	      {
		class="namespace",
		summary="Namespace which contains all loss functions",
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.__base__",
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

april_set_doc("ann.loss.__base__.loss",
	      {
		class="method",
		summary="Computes the loss between two tokens (input and target)",
		description={
		  "The loss is computed for a given input and target tokens.",
		  "This method returns a matrix with the loss for every given pair of patterns,",
		  "and accumulates the loss in its internal state.",
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

april_set_doc("ann.loss.__base__.gradient",
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

april_set_doc("ann.loss.__base__.get_accum_loss",
	      {
		class="method",
		summary="Returns the mean loss from the last reset call",
		outputs = {
		  "The mean loss",
		  "The sample variance of the loss",
		},
	      })


-------------------------------------------------------------------

april_set_doc("ann.loss.__base__.reset",
	      {
		class="method",
		summary="Sets to zero the accumulated loss",
	      })

-------------------------------------------------------------------

april_set_doc("ann.loss.__base__.clone",
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

april_set_doc("ann.loss.local_fmeasure",
	      {
		class="class",
		summary="The FMeasure computed to a bunch of patterns",
		description={
		  "This loss function computes",
		  "FMeasure for a given bunch of patterns.",
		  "Currently it is only implemented to work with one output",
		  "unit models.",
		}
	      })

april_set_doc("ann.loss.local_fmeasure.__call",
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
		outputs={ "An instance of ann.loss.local_fmeasure" },
	      })

