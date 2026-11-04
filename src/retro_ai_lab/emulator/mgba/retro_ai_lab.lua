-- Retro AI Lab bridge for mGBA (0.10 or later).
--
-- Open this script in mGBA (Tools > Scripting... > File > Load script) with
-- a game loaded, then press Connect in Retro AI Lab. The script listens on
-- 127.0.0.1:43110 and answers the app's requests, one JSON object per line
-- (see ARCHITECTURE.md section 5 in the app's repository).
--
-- mGBA's scripting API has no pause, so the game keeps running between the
-- app's requests ("free run", section 5.3); a `step` of N frames is answered
-- once N frames have passed with the asked buttons held.
--
-- Nothing here changes your game, saves or settings. Close the script (or
-- mGBA) to stop it.

local SCRIPT_VERSION = 1
local HOST = "127.0.0.1"
local PORT = tonumber(RETRO_AI_LAB_PORT) or 43110

-------------------------------------------------------------------- base64

local base64 = {}
do
  local alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
  local decode_map = {}
  for i = 1, 64 do
    decode_map[alphabet:sub(i, i)] = i - 1
  end

  function base64.encode(data)
    local out = {}
    local n = #data
    for i = 1, n, 3 do
      local a, b, c = data:byte(i, i + 2)
      local v = a * 65536 + (b or 0) * 256 + (c or 0)
      local c1 = math.floor(v / 262144) % 64
      local c2 = math.floor(v / 4096) % 64
      local c3 = math.floor(v / 64) % 64
      local c4 = v % 64
      out[#out + 1] = alphabet:sub(c1 + 1, c1 + 1) .. alphabet:sub(c2 + 1, c2 + 1)
        .. (b and alphabet:sub(c3 + 1, c3 + 1) or "=")
        .. (c and alphabet:sub(c4 + 1, c4 + 1) or "=")
    end
    return table.concat(out)
  end

  function base64.decode(text)
    text = text:gsub("[^%w%+/=]", "")
    local out = {}
    for i = 1, #text, 4 do
      local q = text:sub(i, i + 3)
      local c1 = decode_map[q:sub(1, 1)]
      local c2 = decode_map[q:sub(2, 2)]
      local c3 = decode_map[q:sub(3, 3)]
      local c4 = decode_map[q:sub(4, 4)]
      if c1 == nil or c2 == nil then
        error("Bad base64.", 0)
      end
      local v = c1 * 262144 + c2 * 4096 + (c3 or 0) * 64 + (c4 or 0)
      local b1 = math.floor(v / 65536) % 256
      local b2 = math.floor(v / 256) % 256
      local b3 = v % 256
      if c3 == nil then
        out[#out + 1] = string.char(b1)
      elseif c4 == nil then
        out[#out + 1] = string.char(b1, b2)
      else
        out[#out + 1] = string.char(b1, b2, b3)
      end
    end
    return table.concat(out)
  end
end

---------------------------------------------------------------------- json

local json = {}
do
  local array_mt = { __jsonarray = true }

  -- Mark a table as a JSON array (so an empty one encodes as [] not {}).
  function json.array(t)
    return setmetatable(t or {}, array_mt)
  end

  local escapes = {
    ['"'] = '\\"', ["\\"] = "\\\\", ["\b"] = "\\b", ["\f"] = "\\f",
    ["\n"] = "\\n", ["\r"] = "\\r", ["\t"] = "\\t",
  }

  -- Escape exactly the bytes JSON requires (controls, quote, backslash) and
  -- DEL. Not %c: on Windows that class also matches bytes of UTF-8 sequences.
  local function encode_string(s)
    local escaped = s:gsub('[\0-\31\127"\\]', function(c)
      return escapes[c] or string.format("\\u%04x", c:byte())
    end)
    return '"' .. escaped .. '"'
  end

  local function is_array(t)
    local mt = getmetatable(t)
    if mt and mt.__jsonarray then
      return true
    end
    if #t == 0 then
      return false
    end
    local count = 0
    for _ in pairs(t) do
      count = count + 1
    end
    return count == #t
  end

  local encode_value

  local function encode_table(t)
    local parts = {}
    if is_array(t) then
      for i = 1, #t do
        parts[i] = encode_value(t[i])
      end
      return "[" .. table.concat(parts, ",") .. "]"
    end
    local keys = {}
    for k in pairs(t) do
      if type(k) ~= "string" then
        error("JSON object keys must be strings.", 0)
      end
      keys[#keys + 1] = k
    end
    table.sort(keys)
    for i, k in ipairs(keys) do
      parts[i] = encode_string(k) .. ":" .. encode_value(t[k])
    end
    return "{" .. table.concat(parts, ",") .. "}"
  end

  encode_value = function(v)
    local kind = type(v)
    if v == nil then
      return "null"
    elseif kind == "boolean" then
      return v and "true" or "false"
    elseif kind == "number" then
      if v ~= v or v == math.huge or v == -math.huge then
        error("JSON can't hold NaN or infinity.", 0)
      end
      if math.type and math.type(v) == "integer" then
        return string.format("%d", v)
      elseif v == math.floor(v) and math.abs(v) < 2 ^ 53 then
        return string.format("%d", v)
      end
      return string.format("%.17g", v)
    elseif kind == "string" then
      return encode_string(v)
    elseif kind == "table" then
      return encode_table(v)
    end
    error("JSON can't hold a " .. kind .. ".", 0)
  end

  json.encode = encode_value

  -- Decoder: recursive descent over the string.
  local function utf8_char(code)
    if code < 0x80 then
      return string.char(code)
    elseif code < 0x800 then
      return string.char(0xC0 + math.floor(code / 64), 0x80 + code % 64)
    elseif code < 0x10000 then
      return string.char(0xE0 + math.floor(code / 4096), 0x80 + math.floor(code / 64) % 64,
        0x80 + code % 64)
    end
    return string.char(0xF0 + math.floor(code / 262144), 0x80 + math.floor(code / 4096) % 64,
      0x80 + math.floor(code / 64) % 64, 0x80 + code % 64)
  end

  local unescapes = {
    ['"'] = '"', ["\\"] = "\\", ["/"] = "/", b = "\b", f = "\f", n = "\n", r = "\r", t = "\t",
  }

  local function skip_space(s, i)
    local _, j = s:find("^[ \t\r\n]*", i)
    return j + 1
  end

  local decode_value

  local function decode_string(s, i)
    -- s:sub(i, i) is the opening quote
    local out = {}
    i = i + 1
    while true do
      local c = s:sub(i, i)
      if c == "" then
        error("Unterminated JSON string.", 0)
      elseif c == '"' then
        return table.concat(out), i + 1
      elseif c == "\\" then
        local e = s:sub(i + 1, i + 1)
        if e == "u" then
          local hex = s:sub(i + 2, i + 5)
          if not hex:match("^%x%x%x%x$") then
            error("Bad \\u escape in JSON string.", 0)
          end
          local code = tonumber(hex, 16)
          i = i + 6
          if code >= 0xD800 and code <= 0xDBFF and s:sub(i, i + 1) == "\\u" then
            local low = tonumber(s:sub(i + 2, i + 5), 16)
            if low and low >= 0xDC00 and low <= 0xDFFF then
              code = 0x10000 + (code - 0xD800) * 1024 + (low - 0xDC00)
              i = i + 6
            end
          end
          out[#out + 1] = utf8_char(code)
        elseif unescapes[e] then
          out[#out + 1] = unescapes[e]
          i = i + 2
        else
          error("Bad escape in JSON string.", 0)
        end
      else
        local j = s:find('["\\]', i) or (#s + 1)
        out[#out + 1] = s:sub(i, j - 1)
        i = j
      end
    end
  end

  local function decode_number(s, i)
    local j = s:find("[^%-%+%.%deE]", i) or (#s + 1)
    local text = s:sub(i, j - 1)
    local n = tonumber(text)
    if n == nil then
      error("Bad JSON number '" .. text .. "'.", 0)
    end
    if math.tointeger and text:match("^%-?%d+$") then
      n = math.tointeger(n) or n
    end
    return n, j
  end

  decode_value = function(s, i)
    i = skip_space(s, i)
    local c = s:sub(i, i)
    if c == "{" then
      local obj = {}
      i = skip_space(s, i + 1)
      if s:sub(i, i) == "}" then
        return obj, i + 1
      end
      while true do
        i = skip_space(s, i)
        if s:sub(i, i) ~= '"' then
          error("Expected a string key in JSON object.", 0)
        end
        local key
        key, i = decode_string(s, i)
        i = skip_space(s, i)
        if s:sub(i, i) ~= ":" then
          error("Expected ':' in JSON object.", 0)
        end
        local value
        value, i = decode_value(s, i + 1)
        obj[key] = value
        i = skip_space(s, i)
        local d = s:sub(i, i)
        if d == "}" then
          return obj, i + 1
        elseif d ~= "," then
          error("Expected ',' or '}' in JSON object.", 0)
        end
        i = i + 1
      end
    elseif c == "[" then
      local arr = json.array({})
      i = skip_space(s, i + 1)
      if s:sub(i, i) == "]" then
        return arr, i + 1
      end
      while true do
        local value
        value, i = decode_value(s, i)
        arr[#arr + 1] = value
        i = skip_space(s, i)
        local d = s:sub(i, i)
        if d == "]" then
          return arr, i + 1
        elseif d ~= "," then
          error("Expected ',' or ']' in JSON array.", 0)
        end
        i = i + 1
      end
    elseif c == '"' then
      return decode_string(s, i)
    elseif s:sub(i, i + 3) == "true" then
      return true, i + 4
    elseif s:sub(i, i + 4) == "false" then
      return false, i + 5
    elseif s:sub(i, i + 3) == "null" then
      return nil, i + 4
    elseif c:match("[%-%d]") then
      return decode_number(s, i)
    end
    error("Unexpected character '" .. c .. "' in JSON.", 0)
  end

  function json.decode(s)
    local value, i = decode_value(s, 1)
    i = skip_space(s, i)
    if i <= #s then
      error("Trailing characters after JSON value.", 0)
    end
    return value
  end
end

------------------------------------------------------------------ emulator

------------------------------------------------------------------ emulator

local server = nil        -- the listening socket
local app = nil           -- the connected app, or nil
local pending = ""        -- a partial line received so far
local closing = false
local key_ids = {}        -- button name -> key index, from C.GBA_KEY / C.GB_KEY
local button_list = {}    -- button names, sorted
local key_mask = 0        -- what is held, as mGBA wants it
local domain_sizes = {}   -- memory domain key -> size, from the handshake
local domain_bases = {}   -- memory domain key -> bus address of its first byte
local states = {}         -- slot name -> state buffer
local waiting = nil       -- a `step` in progress: { id, req, remaining, seen }
local watching = false    -- a watched step leaves the pad to the person (T-887)

local function platform_name()
  local ok, platform = pcall(function() return emu:platform() end)
  if ok and C and C.PLATFORM then
    for name, value in pairs(C.PLATFORM) do
      if value == platform then
        return name
      end
    end
  end
  return "GBA"
end

local function key_table()
  local keys = (platform_name() == "GBA") and C.GBA_KEY or C.GB_KEY
  return keys or C.GBA_KEY or {}
end

local function button_names()
  key_ids = {}
  button_list = {}
  for name, index in pairs(key_table()) do
    if type(name) == "string" and type(index) == "number" then
      key_ids[name] = index
      button_list[#button_list + 1] = name
    end
  end
  table.sort(button_list)
  key_mask = 0
  return json.array(button_list)
end

local function domains()
  local out = json.array({})
  domain_sizes = {}
  domain_bases = {}
  local keys = {}
  for key, _ in pairs(emu.memory) do
    keys[#keys + 1] = key
  end
  table.sort(keys)
  for _, key in ipairs(keys) do
    local domain = emu.memory[key]
    local ok, size = pcall(function() return domain:size() end)
    local okb, base = pcall(function() return domain:base() end)
    if ok and type(size) == "number" and size > 0 then
      out[#out + 1] = { name = key, size = size }
      domain_sizes[key] = size
      domain_bases[key] = (okb and type(base) == "number") and base or 0
    end
  end
  return out
end

-- Bytes [addr, addr + len) of a memory domain as a Lua string.
local function read_bytes(domain, addr, len)
  local size = domain_sizes[domain]
  if size == nil then
    error("Unknown memory domain '" .. tostring(domain) .. "'.", 0)
  end
  if math.type(addr) ~= "integer" or math.type(len) ~= "integer" or addr < 0 or len < 0 then
    error("A read needs a whole-number address and length.", 0)
  end
  if addr + len > size then
    error("Read of " .. len .. " bytes at " .. addr .. " is outside '" .. domain
      .. "' (size " .. size .. ").", 0)
  end
  if len == 0 then
    return ""
  end
  local data = emu.memory[domain]:readRange(domain_bases[domain] + addr, len)
  if type(data) ~= "string" or #data ~= len then
    error("mGBA returned " .. (type(data) == "string" and #data or "no") .. " bytes for a read of "
      .. len .. ".", 0)
  end
  return data
end

local function serve_reads(reads)
  if type(reads) ~= "table" then
    error("'reads' must be a list of {domain, addr, len}.", 0)
  end
  local out = json.array({})
  for i, item in ipairs(reads) do
    if type(item) ~= "table" then
      error("'reads' must be a list of {domain, addr, len}.", 0)
    end
    out[i] = base64.encode(read_bytes(item.domain, item.addr, item.len))
  end
  return out
end

local function set_held(buttons)
  if type(buttons) ~= "table" then
    error("'buttons' must be a table of button names to true or false.", 0)
  end
  local mask = 0
  for name, value in pairs(buttons) do
    if type(name) ~= "string" or key_ids[name] == nil then
      error("This pad has no button '" .. tostring(name) .. "'.", 0)
    end
    if type(value) ~= "boolean" then
      error("Button '" .. name .. "' must be true or false.", 0)
    end
    if value then
      mask = mask | (1 << key_ids[name])
    end
  end
  key_mask = mask
end

local function apply_held()
  emu:setKeys(key_mask)
end

-- The names of the buttons a key mask holds, sorted.
local function mask_names(mask)
  local names = {}
  for name, index in pairs(key_ids) do
    if mask & (1 << index) ~= 0 then
      names[#names + 1] = name
    end
  end
  table.sort(names)
  return json.array(names)
end

local function screenshot_path()
  local dir = os.getenv("TEMP") or os.getenv("TMP") or os.getenv("TMPDIR")
  if dir and dir ~= "" then
    return dir .. "/retro_ai_lab_frame.png"
  end
  return os.tmpname() .. ".png"
end

local function take_screenshot()
  local path = screenshot_path()
  os.remove(path)
  emu:screenshot(path)
  local file = io.open(path, "rb")
  if file == nil then
    error("mGBA did not write a screenshot.", 0)
  end
  local data = file:read("*a")
  file:close()
  os.remove(path)
  if data == nil or #data == 0 then
    error("mGBA wrote an empty screenshot.", 0)
  end
  return base64.encode(data)
end

-- The app on the same PC can name a file instead (T-825): the screenshot is
-- written there and only the path goes back, sparing the base64 round trip.
local function image_file(req)
  local path = req.image_file
  if path == nil then return nil end
  if type(path) ~= "string" or path == "" or path:find("[\r\n]") or path:sub(-4) ~= ".png" then
    error("'image_file' must be a path ending in .png.", 0)
  end
  return path
end

local function screenshot_into(path)
  os.remove(path)
  emu:screenshot(path)
  local file = io.open(path, "rb")
  if file == nil then
    error("mGBA did not write a screenshot.", 0)
  end
  file:close()
  return path
end

local function screenshot_body(req, body)
  local path = image_file(req)
  if path then
    body.image_file = screenshot_into(path)
  else
    body.image = take_screenshot()
  end
  return body
end

local function slot_name(slot)
  if type(slot) ~= "string" or slot == "" or slot:find("[^%w_]") then
    error("'slot' must be a name of letters, digits and underscores.", 0)
  end
  return slot
end

local handlers = {}

local function command_names()
  local names = json.array({})
  for name in pairs(handlers) do
    names[#names + 1] = name
  end
  table.sort(names)
  return names
end

local function crc32_text()
  local ok, crc = pcall(function() return emu:checksum(C.CHECKSUM.CRC32) end)
  if ok and type(crc) == "number" then
    return string.format("%08X", crc & 0xFFFFFFFF)
  end
  if ok and type(crc) == "string" then
    local hex = {}
    for i = 1, #crc do
      hex[#hex + 1] = string.format("%02X", crc:byte(i))
    end
    return table.concat(hex)
  end
  return ""
end

function handlers.hello(req)
  local title = emu:getGameTitle()
  local code = ""
  local ok, got = pcall(function() return emu:getGameCode() end)
  if ok and type(got) == "string" then
    code = got
  end
  return {
    script = SCRIPT_VERSION,
    emulator = "mGBA",
    system = platform_name(),
    rom = (type(title) == "string" and title ~= "") and title or code,
    code = code,
    hash = crc32_text(),
    domains = domains(),
    buttons = button_names(),
    commands = command_names(),
    hold = false,
  }
end

function handlers.ping(req)
  return {}
end

function handlers.input(req)
  set_held(req.buttons)
  apply_held()
  return {}
end

-- mGBA can't pause from a script: free run is the only mode, so this is a
-- no-op that keeps the command set the same as BizHawk's.
function handlers.mode(req)
  if type(req.free_run) ~= "boolean" then
    error("'free_run' must be true or false.", 0)
  end
  return {}
end

function handlers.screenshot(req)
  return screenshot_body(req, {})
end

function handlers.read(req)
  return { reads = serve_reads(req.reads) }
end

local MAX_STEP_FRAMES = 3600

-- Hold the pad and answer once the frames have passed (see on_frame).
function handlers.step(req)
  local frames = req.frames
  if math.type(frames) ~= "integer" or frames < 1 or frames > MAX_STEP_FRAMES then
    error("'frames' must be a whole number from 1 to " .. MAX_STEP_FRAMES .. ".", 0)
  end
  if req.buttons ~= nil then
    set_held(req.buttons)
    apply_held()
  end
  if req.reads ~= nil and type(req.reads) ~= "table" then
    error("'reads' must be a list of {domain, addr, len}.", 0)
  end
  if req.screenshot ~= nil and type(req.screenshot) ~= "boolean" then
    error("'screenshot' must be true or false.", 0)
  end
  if req.watch ~= nil and type(req.watch) ~= "boolean" then
    error("'watch' must be true or false.", 0)
  end
  if req.watch and req.buttons ~= nil then
    error("A step cannot hold the pad and watch it at once.", 0)
  end
  image_file(req)  -- refuse a bad path now, not after the frames have passed
  watching = req.watch == true
  if watching then
    key_mask = 0
  end
  waiting = { id = req.id, req = req, remaining = frames, seen = watching and 0 or nil }
  return nil  -- answered later
end

function handlers.save(req)
  local name = slot_name(req.slot)
  local buffer = emu:saveStateBuffer()
  if type(buffer) ~= "string" or #buffer == 0 then
    error("mGBA did not produce a save state.", 0)
  end
  states[name] = buffer
  return {}
end

function handlers.load(req)
  local name = slot_name(req.slot)
  local buffer = states[name]
  if buffer == nil then
    return { loaded = false }
  end
  local ok = emu:loadStateBuffer(buffer)
  return { loaded = ok ~= false }
end

function handlers.bye(req)
  closing = true
  return {}
end

--------------------------------------------------------------- connection

local function send(body)
  if app then
    app:send(json.encode(body) .. "\n")
  end
end

local function fail(id, message)
  send({ id = id, ok = false, error = message })
end

local function reply(id, body)
  body.id = id
  body.ok = true
  body.frame = emu:currentFrame()
  send(body)
end

local function drop_app()
  if app then
    app:close()
    app = nil
  end
  pending = ""
  closing = false
  waiting = nil
  watching = false
  key_mask = 0
  if next(key_ids) ~= nil then
    apply_held()
  end
end

local function finish_step()
  local step = waiting
  waiting = nil
  local ok, body = pcall(function()
    local out = {}
    if step.req.reads ~= nil then
      out.reads = serve_reads(step.req.reads)
    end
    if step.req.screenshot then
      screenshot_body(step.req, out)
    end
    if step.seen ~= nil then
      out.pressed = mask_names(step.seen)
    end
    return out
  end)
  if ok then
    reply(step.id, body)
  else
    fail(step.id, tostring(body))
  end
end

local function serve_line(line)
  local ok, req = pcall(json.decode, line)
  if not ok or type(req) ~= "table" then
    fail(nil, "That was not a JSON object.")
    return
  end
  if waiting ~= nil then
    fail(req.id, "A step is still running; wait for its answer.")
    return
  end
  local handler = handlers[req.cmd]
  if handler == nil then
    fail(req.id, "Unknown command '" .. tostring(req.cmd) .. "'.")
    return
  end
  local handled, body = pcall(handler, req)
  if not handled then
    fail(req.id, tostring(body))
    return
  end
  if body ~= nil then
    reply(req.id, body)
  end
  if closing then
    drop_app()
  end
end

local function on_received()
  if app == nil then
    return
  end
  local chunk, err = app:receive(65536)
  if chunk == nil or chunk == "" then
    if err ~= nil and err ~= "timeout" and err ~= 0 then
      console:log("Retro AI Lab disconnected (" .. tostring(err) .. ").")
      drop_app()
    end
    return
  end
  pending = pending .. chunk
  while app do
    local cut = pending:find("\n", 1, true)
    if cut == nil then
      return
    end
    local line = pending:sub(1, cut - 1):gsub("\r$", "")
    pending = pending:sub(cut + 1)
    serve_line(line)
  end
end

local function on_error(err)
  console:log("Retro AI Lab link error (" .. tostring(err) .. ").")
  drop_app()
end

local function on_accept()
  local candidate = server:accept()
  if candidate == nil then
    return
  end
  if app ~= nil then
    candidate:close()  -- one app at a time
    return
  end
  app = candidate
  pending = ""
  app:add("received", on_received)
  app:add("error", on_error)
  console:log("Retro AI Lab connected.")
end

-- While watching, the keys are left to the person's controller (T-887).
local function on_keys_read()
  if app and not watching then
    apply_held()
  end
end

local function on_frame()
  if waiting == nil then
    return
  end
  if waiting.seen ~= nil then
    waiting.seen = waiting.seen | emu:getKeys()
  end
  waiting.remaining = waiting.remaining - 1
  if waiting.remaining <= 0 then
    finish_step()
  end
end

local function main()
  local err
  server, err = socket.bind(HOST, PORT)
  if server == nil then
    error("Retro AI Lab could not listen on port " .. PORT .. ": " .. tostring(err), 0)
  end
  local ok, listen_err = server:listen(1)
  if ok == false then
    error("Retro AI Lab could not listen on port " .. PORT .. ": " .. tostring(listen_err), 0)
  end
  server:add("received", on_accept)
  server:add("error", on_error)
  callbacks:add("keysRead", on_keys_read)
  callbacks:add("frame", on_frame)
  callbacks:add("stop", function()
    drop_app()
    if server then
      server:close()
    end
  end)
  console:log("Retro AI Lab is waiting for the app on port " .. tostring(PORT) .. ".")
end

-- Tests run the pieces without an emulator: when a host provides this
-- table, the script hands over its internals instead of starting.
if RETRO_AI_LAB_INTERNALS then
  RETRO_AI_LAB_INTERNALS.json = json
  RETRO_AI_LAB_INTERNALS.base64 = base64
  RETRO_AI_LAB_INTERNALS.main = main
  RETRO_AI_LAB_INTERNALS.version = SCRIPT_VERSION
else
  main()
end
