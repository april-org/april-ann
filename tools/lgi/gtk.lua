-- module gtk
local gtk = {}

local lgi = require "lgi"
local Gtk = lgi.Gtk

function gtk.show(param)
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
  local window = Gtk.Window {
    title = 'Prueba',
    Gtk.Box {
      id = 'vbox',
      orientation = 'VERTICAL',
      spacing = 8,
      border_width = 8,
      Gtk.Frame {
	shadow_type = 'IN',
	halign = 'CENTER',
	valign = 'CENTER',
	Gtk.Image { file = param, },
      },
      Gtk.ToggleButton {
	id = 'exit',
	label = 'exit',
      },
    },
  }
  function window.child.exit:on_toggled()
    Gtk.main_quit()
  end
  window:show_all()
  if tmp_file then os.remove(tmp_file) end
  Gtk.main()
end

return gtk
