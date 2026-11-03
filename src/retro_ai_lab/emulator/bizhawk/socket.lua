-- Retro AI Lab: LuaSocket loader for BizHawk (EmuHawk).
--
-- EmuHawk's Lua has no socket library of its own, so `require("socket")`
-- fails with "module 'socket' not found". This file sits next to
-- retro_ai_lab.lua and loads the LuaSocket core from luasocket/<arch>/
-- (socket-windows-5-4.dll for a 64-bit EmuHawk 2.9+), then adds the two
-- helpers the bridge needs. Keep the two files and the luasocket folder
-- together.

local function script_dir()
  local info = debug and debug.getinfo and debug.getinfo(1, "S")
  local source = info and info.source or ""
  if source:sub(1, 1) == "@" then
    local dir = source:sub(2):match("^(.*)[/\\]")
    if dir then return dir end
  end
  if package.config:sub(1, 1) == "\\" and io.popen then
    local pwd = io.popen("cd"):read("*l")
    if pwd and pwd ~= "" then return pwd end
  end
  return "."
end

local function lua_version()
  local major, minor = _VERSION:match("Lua (%d+)%.(%d+)")
  if tonumber(major) == 5 and tonumber(minor) >= 4 then return "5-4" end
  return "5-1"
end

local function platform()
  if package.config:sub(1, 1) == "\\" then
    local arch = os.getenv("PROCESSOR_ARCHITECTURE") or ""
    local wow = os.getenv("PROCESSOR_ARCHITEW6432") or ""
    local bits = (arch:find("64") or wow:find("64")) and "x64" or "x86"
    return "windows", "dll", bits
  end
  return "linux", "so", "x64"
end

local function load_core()
  local ok, core = pcall(require, "socket.core")
  if ok and type(core) == "table" then return core end
  local os_name, ext, arch = platform()
  local path = script_dir() .. "/luasocket/" .. arch .. "/socket-" .. os_name .. "-"
    .. lua_version() .. "." .. ext
  local open, err = package.loadlib(path, "luaopen_socket_core")
  if not open then
    error("Retro AI Lab could not load LuaSocket from " .. path .. " (" .. tostring(err)
      .. "). Keep retro_ai_lab.lua and its luasocket folder together, and use a 64-bit "
      .. "BizHawk 2.9 or newer.", 0)
  end
  return open()
end

local core = load_core()
local M = {}
for key, value in pairs(core) do M[key] = value end
M.core = core

function M.connect(address, port, laddress, lport)
  local sock, err = core.tcp()
  if not sock then return nil, err end
  if laddress then
    local res, bind_err = sock:bind(laddress, lport, -1)
    if not res then return nil, bind_err end
  end
  local res, connect_err = sock:connect(address, port)
  if not res then return nil, connect_err end
  return sock
end

function M.bind(host, port, backlog)
  local sock, err = core.tcp()
  if not sock then return nil, err end
  sock:setoption("reuseaddr", true)
  local res, bind_err = sock:bind(host, port)
  if not res then return nil, bind_err end
  local listening, listen_err = sock:listen(backlog or 32)
  if not listening then return nil, listen_err end
  return sock
end

package.loaded["socket"] = M
return M
