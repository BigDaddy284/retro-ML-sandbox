-- Retro AI Lab bridge for BizHawk (EmuHawk).
--
-- Open this script in EmuHawk (Tools > Lua Console > Script > Open Script)
-- with a game loaded, then press Connect in Retro AI Lab. The script listens
-- on 127.0.0.1:43110 and answers the app's requests, one JSON object per
-- line (see ARCHITECTURE.md section 5 in the app's repository).
--
-- Nothing here changes your game, saves or settings. Close the script (or
-- EmuHawk) to stop it.

local SCRIPT_VERSION = 1
local HOST = "127.0.0.1"
local PORT = tonumber(RETRO_AI_LAB_PORT) or 43110

-- EmuHawk has no socket library; socket.lua next to this file loads the one
-- in the luasocket folder. Put this file's folder first so require finds it
-- even when EmuHawk's working directory is elsewhere.
do
  local info = debug and debug.getinfo and debug.getinfo(1, "S")
  local source = info and info.source or ""
  local dir = source:sub(1, 1) == "@" and source:sub(2):match("^(.*)[/\\]") or nil
  if dir then package.path = dir .. "/?.lua;" .. package.path end
end
local socket = require("socket")

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

-- What the app can ask for, by command name. A handler returns the body of
-- the answer (a table) or raises an error with the message for the app.

local server = nil   -- the listening socket
local app = nil      -- the connected app, or nil
local pending = ""   -- a partial line received so far
local free_run = false
local closing = false
local pad = {}       -- button name -> true, from the handshake
local domain_sizes = {}  -- memory domain name -> size, from the handshake
local prefixed = false  -- true when the core names buttons "P1 A" rather than "A"
local held = {}      -- the full pad table joypad.set gets before every frame

local function unpause()
  if client and client.unpause then
    client.unpause()
  end
end

local function pause()
  if client and client.pause then
    client.pause()
  end
end

local function domains()
  local list = {}
  local names = memory.getmemorydomainlist()
  if type(names) == "string" then
    for name in names:gmatch("[^\n]+") do
      list[#list + 1] = name
    end
  elseif type(names) == "table" then
    for _, name in pairs(names) do
      if type(name) == "string" and name ~= "" then
        list[#list + 1] = name
      end
    end
  end
  local out = json.array({})
  domain_sizes = {}
  for _, name in ipairs(list) do
    local ok, size = pcall(memory.getmemorydomainsize, name)
    if ok and type(size) == "number" and size > 0 then
      -- A domain the core lists but cannot peek (a .NET exception on the
      -- first byte) is left out, so the app never asks for it (T-832).
      local readable, err = pcall(memory.read_bytes_as_array, 0, 1, name)
      if readable then
        out[#out + 1] = { name = name, size = size }
        domain_sizes[name] = size
      else
        print("Retro AI Lab: not offering memory domain '" .. name
          .. "' because the emulator cannot read it: " .. tostring(err))
      end
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
  local ok, values = pcall(memory.read_bytes_as_array, addr, len, domain)
  if not ok then
    error("The emulator could not read " .. len .. " bytes at " .. addr .. " of '"
      .. domain .. "': " .. tostring(values), 0)
  end
  local parts = {}
  local step = 4096  -- table.unpack is limited by the C stack
  for i = 1, len, step do
    parts[#parts + 1] = string.char(table.unpack(values, i, math.min(i + step - 1, len)))
  end
  return table.concat(parts)
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

-- Only pad entries that are pressed-or-not count as buttons; analog axes
-- (numbers) are left out. Remembers the pad for input.
local function button_names()
  local names = {}
  prefixed = false
  local ok, table_ = pcall(joypad.get, 1)
  if ok and type(table_) == "table" then
    for name, value in pairs(table_) do
      if type(value) == "boolean" then
        names[#names + 1] = name
      end
    end
  end
  if #names == 0 then
    ok, table_ = pcall(joypad.get)
    if ok and type(table_) == "table" then
      for name, value in pairs(table_) do
        if type(value) == "boolean" and name:sub(1, 3) == "P1 " then
          names[#names + 1] = name:sub(4)
          prefixed = true
        end
      end
    end
  end
  table.sort(names)
  pad = {}
  held = {}
  for _, name in ipairs(names) do
    pad[name] = true
    held[name] = false
  end
  return json.array(names)
end

-- Hold the current button table for the next frame.
local function apply_held()
  if next(pad) == nil then
    return
  end
  if prefixed then
    local full = {}
    for name, value in pairs(held) do
      full["P1 " .. name] = value
    end
    joypad.set(full)
  else
    joypad.set(held, 1)
  end
end

local function release_all()
  for name in pairs(held) do
    held[name] = false
  end
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

function handlers.hello(req)
  if not free_run then
    pause()
  end
  return {
    script = SCRIPT_VERSION,
    bizhawk = client.getversion(),
    system = emu.getsystemid(),
    rom = gameinfo.getromname(),
    hash = gameinfo.getromhash(),
    domains = domains(),
    buttons = button_names(),
    commands = command_names(),
  }
end

function handlers.ping(req)
  return {}
end

local function set_held(buttons)
  if type(buttons) ~= "table" then
    error("'buttons' must be a table of button names to true or false.", 0)
  end
  for name, value in pairs(buttons) do
    if type(name) ~= "string" or not pad[name] then
      error("This pad has no button '" .. tostring(name) .. "'.", 0)
    end
    if type(value) ~= "boolean" then
      error("Button '" .. name .. "' must be true or false.", 0)
    end
  end
  release_all()
  for name, value in pairs(buttons) do
    held[name] = value
  end
end

function handlers.input(req)
  set_held(req.buttons)
  return {}
end

function handlers.mode(req)
  if type(req.free_run) ~= "boolean" then
    error("'free_run' must be true or false.", 0)
  end
  free_run = req.free_run
  if free_run then
    unpause()
  else
    pause()
  end
  return {}
end

-- Where screenshots go on their way to the app: the user's temp folder.
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
  client.screenshot(path)
  local file = io.open(path, "rb")
  if file == nil then
    error("EmuHawk did not write a screenshot.", 0)
  end
  local data = file:read("*a")
  file:close()
  os.remove(path)
  if data == nil or #data == 0 then
    error("EmuHawk wrote an empty screenshot.", 0)
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
  client.screenshot(path)
  local file = io.open(path, "rb")
  if file == nil then
    error("EmuHawk did not write a screenshot.", 0)
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

function handlers.screenshot(req)
  return screenshot_body(req, {})
end

function handlers.read(req)
  return { reads = serve_reads(req.reads) }
end

local MAX_STEP_FRAMES = 3600

-- Run exactly `frames` frames with the held pad, then stop again in hold mode.
--
-- EmuHawk never wakes a script waiting in emu.frameadvance() while it is
-- paused (MainForm.StepRunLoop_Core runs a frame only when unpaused or on
-- the user's Frame Advance key), so hold mode unpauses first and pauses
-- again after the last frame. The pad is latched into the controller at the
-- top of each turn of EmuHawk's loop and the overrides are cleared before
-- the frame runs, so a pad set from the paused loop must be followed by one
-- emu.yield() before the first frame, and set again between frames.
local function run_frames(frames)
  if free_run then
    for _ = 1, frames do
      apply_held()
      emu.frameadvance()
    end
    return
  end
  apply_held()
  emu.yield()
  unpause()
  for i = 1, frames do
    emu.frameadvance()
    if i < frames then
      apply_held()
    end
  end
  pause()
end

-- Advance the game: hold the pad, run the frames, then report what was asked.
function handlers.step(req)
  local frames = req.frames
  if math.type(frames) ~= "integer" or frames < 1 or frames > MAX_STEP_FRAMES then
    error("'frames' must be a whole number from 1 to " .. MAX_STEP_FRAMES .. ".", 0)
  end
  if req.buttons ~= nil then
    set_held(req.buttons)
  end
  if req.reads ~= nil and type(req.reads) ~= "table" then
    error("'reads' must be a list of {domain, addr, len}.", 0)
  end
  if req.screenshot ~= nil and type(req.screenshot) ~= "boolean" then
    error("'screenshot' must be true or false.", 0)
  end
  run_frames(frames)
  local body = {}
  if req.reads ~= nil then
    body.reads = serve_reads(req.reads)
  end
  if req.screenshot then
    screenshot_body(req, body)
  end
  return body
end

function handlers.speed(req)
  local percent = req.percent
  if type(percent) ~= "number" or percent ~= percent or percent <= 0 then
    error("'percent' must be a number above zero.", 0)
  end
  client.speedmode(math.max(1, math.floor(percent + 0.5)))
  return {}
end

local function state_path(slot)
  if type(slot) ~= "string" or slot == "" or slot:find("[^%w_]") then
    error("'slot' must be a name of letters, digits and underscores.", 0)
  end
  local dir = os.getenv("TEMP") or os.getenv("TMP") or os.getenv("TMPDIR") or "."
  return dir .. "/retro_ai_lab_" .. slot .. ".State"
end

function handlers.save(req)
  savestate.save(state_path(req.slot), true)
  return {}
end

function handlers.load(req)
  local loaded = savestate.load(state_path(req.slot), true)
  return { loaded = loaded == true }
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

local function drop_app()
  if app then
    app:close()
    app = nil
  end
  pending = ""
  free_run = false
  closing = false
  release_all()
  unpause()
end

local function serve_line(line)
  local ok, req = pcall(json.decode, line)
  if not ok or type(req) ~= "table" then
    fail(nil, "That was not a JSON object.")
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
  body.id = req.id
  body.ok = true
  body.frame = emu.framecount()
  send(body)
  if closing then
    drop_app()
  end
end

-- Accept a connection if none, then serve every complete line waiting.
local function service()
  if app == nil then
    local candidate = server:accept()
    if candidate == nil then
      return
    end
    candidate:settimeout(0)
    app = candidate
    pending = ""
    print("Retro AI Lab connected.")
  end
  while app do
    local line, err, partial = app:receive("*l")
    if line then
      serve_line(pending .. line)
      pending = ""
    elseif err == "timeout" then
      if partial and partial ~= "" then
        pending = pending .. partial
      end
      return
    else
      print("Retro AI Lab disconnected (" .. tostring(err) .. ").")
      drop_app()
      return
    end
  end
end

local function main()
  local err
  server, err = socket.bind(HOST, PORT)
  if server == nil then
    error("Retro AI Lab could not listen on port " .. PORT .. ": " .. tostring(err), 0)
  end
  server:settimeout(0)
  local _, bound_port = server:getsockname()
  print("Retro AI Lab is waiting for the app on port " .. tostring(bound_port) .. ".")
  if event and event.onexit then
    event.onexit(function()
      if app then
        app:close()
      end
      if server then
        server:close()
      end
    end)
  end
  while true do
    service()
    if app and not free_run then
      emu.yield()
    else
      if app then
        apply_held()
      end
      emu.frameadvance()
    end
  end
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
