return {
  -- worker data
  name           = "WORKER",
  bind_address   = '*',
  port           = 4000,
  nump           = 4,
  mem            = "4G",
  -- master server data
  master_address = "localhost",
  master_port    = 8888,
}
