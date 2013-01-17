img = Image.load("a01-000u-s00-02.png")

baseline_inf,baseline_sup = img:base_lines()

print("los base lines son ",baseline_inf,baseline_sup)

_,w,h=img:info()
for i=1,w do
  img:putpixel(i-1,baseline_inf,0.5)
  img:putpixel(i-1,baseline_sup,0.6)
end

Image.save(img,"blah.pgm")


