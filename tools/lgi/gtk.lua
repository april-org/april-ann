-- module gtk
local gtk = {}

local lgi = require "lgi"
local Gtk = lgi.Gtk
local GdkPixbuf = lgi.GdkPixbuf

function gtk.show(...)
  local windows = {}
  for i=1,select('#',...) do
    local param = select(i,...)
    local gtk_image = nil
    local tmp_file  = nil
    if type(param) ~= "string" then
      local img = nil
      if type(param) == "Image" or type(param) == "ImageRGB" then
	img = param
      elseif type(param) == "matrix" then
	img = Image(param)
      else
	error("Not supported data type: " .. type(param))
      end
      tmp_file = os.tmpname()
      ImageIO.write(img, tmp_file, 'png')
      param = tmp_file
    end
    local sx,sy  = 1,1
    local pixbuf = GdkPixbuf.Pixbuf.new_from_file(param)
    local window = Gtk.Window {
      title = 'Prueba',
      Gtk.Box {
	id = 'vbox',
	orientation = 'VERTICAL',
	spacing = 8,
	border_width = 8,
	Gtk.Box {
	  id = 'top',
	  orientation = 'HORIZONTAL',
	  spacing = 8,
	  border_width = 8,
	  Gtk.ToggleButton {
	    id = 'fit',
	    label = 'fit',
	  },
	  Gtk.Box {
	    orientation = 'VERTICAL',
	    Gtk.ToggleButton {
	      id = 'zoomIN',
	      label = '+',
	    },
	    Gtk.ToggleButton {
	      id = 'zoomOUT',
	      label = '-',
	    },
	  },
	  Gtk.ToggleButton {
	    id = 'exit',
	    label = 'exit',
	  },
	},
	Gtk.Frame {
	  shadow_type = 'IN',
	  halign = 'CENTER',
	  valign = 'CENTER',
	  Gtk.Image { id='image', pixbuf=pixbuf },
	},
      },
    }
    local function scale()
      local scaled_pixbuf = pixbuf:scale_simple(pixbuf.width  * sx,
						pixbuf.height * sy,
						'BILINEAR')
      window.child.image.pixbuf = scaled_pixbuf
      window.child.image:queue_draw()
      collectgarbage("collect")
    end
    function window.child.exit:on_toggled()
      for i=1,#windows do windows[i]:hide() end
      Gtk.main_quit()
    end
    function window.child.fit:on_toggled()
      local w,h = window.width-40,window.height - window.child.top.height - 50
      local ratio = pixbuf.height/pixbuf.width
      if w*ratio > h then
	sy=h/pixbuf.height
	sx=sy
      else
	sx=w/pixbuf.width
	sx=sy
      end
      scale()
    end
    function window.child.zoomIN:on_toggled()
      sx,sy = sx*1.1,sy*1.1
      scale()
    end
    function window.child.zoomOUT:on_toggled()
      sx,sy = sx*0.9,sy*0.9
      scale()
    end
    window:show_all()
    table.insert(windows, window)
    if tmp_file then os.remove(tmp_file) end
  end
  Gtk.main()
end

return gtk
