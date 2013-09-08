---------------------------
-- BINDING DOCUMENTATION --
---------------------------
april_set_doc("random", {
		class       = "class",
		summary     = "Pseudo-random numbers generator", })

april_set_doc("random.__call", {
		class = "method", summary = "Constructor",
		description ={
		  "Constructor of random objects",
		},
		params = {
		  "A number with the initial seed",
		},
		outputs = { "A random instantiated object" }, })

april_set_doc("random.__call", {
		class = "method", summary = "Constructor",
		description ={
		  "Constructor of random objects",
		},
		params = {
		  "A table with numbers for initial seeds",
		},
		outputs = { "A random instantiated object" }, })

april_set_doc("random.__call", {
		class = "method", summary = "Constructor",
		description ={
		  "Constructor of random objects, the initial seed",
		  "is taken from current time.",
		},
		outputs = { "A random instantiated object" }, })

april_set_doc("random.rand", {
		class = "method",
		summary = "Returns a double precision random number, interval [0,n]",
		params = {
		  "The n number for the interval [optional]. By default is n=1.",
		},
		outputs = { "A double precision random number" }, })

april_set_doc("random.randExc", {
		class = "method",
		summary = "Returns a double precision random number, interval [0,n)",
		params = {
		  "The n number for the interval [optional]. By default is n=1.",
		},
		outputs = { "A double precision random number" }, })

april_set_doc("random.randDblExc", {
		class = "method",
		summary = "Returns a double precision random number, interval (0,n)",
		params = {
		  "The n number for the interval [optional]. By default is n=1.",
		},
		outputs = { "A double precision random number" }, })

april_set_doc("random.randInt", {
		class = "method",
		summary = "Returns an integer random number, interval [0,2^32-1]",
		outputs = { "An integer random number" }, })

april_set_doc("random.randInt", {
		class = "method",
		summary = "Returns an integer random number, interval [0,x]",
		params = {
		  "The x number for the interval.",
		},
		outputs = { "An integer random number" }, })

april_set_doc("random.randInt", {
		class = "method",
		summary = "Returns an integer random number, interval [x,y]",
		params = {
		  "The x number for the interval.",
		  "The y number for the interval.",
		},
		outputs = { "An integer random number" }, })

april_set_doc("random.shuffle", {
		class = "method",
		summary = "Returns a permutation of an array",
		params = {
		  "A number with the size of the sequence",
		},
		outputs = { "A table with the permutation" }, })

april_set_doc("random.shuffle", {
		class = "method",
		summary = "Returns a random sort of an array",
		params = {
		  "A table with the array for reordering",
		},
		outputs = { "A table with the given array randomly reordered" }, })

april_set_doc("random.choose", {
		class = "method",
		summary = "Returns a random position of a given array",
		params = {
		  "A table with the array",
		},
		outputs = { "A number which is a random position in the given array" }, })

april_set_doc("random.randNorm", {
		class = "method",
		summary = "Returns a Gaussian random number",
		params = {
		  "Mean of the Gaussian",
		  "Variance of the Gaussian",
		},
		outputs = { "A Gaussian random number" }, })

april_set_doc("random.seed", {
		class = "method",
		summary = "Sets the current seed.",
		params = {
		  "A number [optional]. By default is taken from current time.",
		}, })

april_set_doc("random.seed", {
		class = "method",
		summary = "Sets the current seed.",
		params = {
		  "A table with an array of seeds.",
		}, })

april_set_doc("random.clone", {
		class = "method",
		summary = "Returns a deep-copy of the caller object",
		outputs = { "An instance of random." }, })

april_set_doc("random.toTable", {
		class = "method",
		summary = "Serializes the random state to a table.",
		outputs = { "A table with the random state." }, })

april_set_doc("random.fromTable", {
		class = "method",
		summary = "Loads the random state from a table.",
		params = { "A table with the random state" }, })

-------------------------------------------------------------------------------

april_set_doc("random.dice", {
		class       = "class",
		summary     = "A class for random dice parametrization", })

april_set_doc("random.dice.__call", {
		class       = "method",
		summary     = "Constructor",
		params = {
		  "A table with the positive score of each side of the dice",
		}, })

april_set_doc("random.dice.outcomes", {
		class       = "method",
		summary     = "Returns the number of sides of the dice",
		outputs = { "The number of sides" }, })

april_set_doc("random.dice.thorwn", {
		class       = "method",
		summary     = "Throws the dice and returns the outcome",
		params = {
		  "A random instance",
		},
		outputs = { "The side of the dice (the outcome)" }, })
