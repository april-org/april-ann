return {
  fixed_params = {
    { option="-x", value=1, tag="x"  },
    { option="-y", value=2, tag="y"  },
  },
  random_params = {
    { option="-a", tag="a", sampling = "uniform", type="integer",
      values= { { min=-5, max=10 }, { min=20, max=100, step=10 } } },
    { option="-b", tag="b", sampling = "uniform", type="real", values= {
	{ min=-5.0, max=10.0}, { min=20.0, max=100.0 } } },
    { option="-c", tag="c", sampling = "uniform", values= { "n", "m", "l", "o" } },
    { option="-d", tag="d", sampling = "gaussian", values= { mean=5, variance=0.1 } },
  },
  script = "",
  exec   = "echo ",
  working_dir = ".",
  --  seed = ANY_SEED_VALUE (if not given, take "echo $RANDOM" as seed)
  n = 2000,
}
