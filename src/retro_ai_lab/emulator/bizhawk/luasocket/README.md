# luasocket

Prebuilt LuaSocket core for EmuHawk's Lua. EmuHawk ships no socket library,
so `socket.lua` (one folder up) loads `x64/socket-windows-5-4.dll` (Lua 5.4,
64-bit EmuHawk 2.9+) or `x64/socket-windows-5-1.dll` (older Lua 5.1 builds)
with `package.loadlib`.

These two files are the binaries the Archipelago multiworld project ships in
`data/lua/x64/` (https://github.com/ArchipelagoMW/Archipelago, MIT licence).
They are unmodified copies of LuaSocket (https://github.com/lunarmodules/luasocket,
MIT licence, Copyright (C) 2004-2022 Diego Nehab). SHA-256 at the time of copying:

- `socket-windows-5-4.dll` 47ebef934e70acfe5bdb438ccbb2c8e3f8ac57d8acc833e5b1bd445ba5e2f378
- `socket-windows-5-1.dll` 6e81f86152d9118a7a61ad9c9ca5465a95fbc33d82b0398dce13d47fbd47c4a7

See `LICENSE` in this folder.
