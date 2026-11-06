# Patches for Hex

`prepare-hex.sh` applies every `*.patch` here to the Hex it builds, and skips
one that no longer applies (usually because Hex fixed the problem its own way).
Delete a patch once every Hex release you build carries the fix.

- `0001-macos-receipt-ignores-volume-number.patch`: Hex 3.2.0 and 3.3.0 check
  an imported model against a file id that includes the volume number
  (`st_dev`), which macOS can change at boot. After a reboot the import reads
  as changed, the listener waits for setup, and the shortcut does nothing
  (`hex model repair <id>` fixes it until the next reboot). This compares only
  the inode half on macOS. Hex dev fixed it on 2026-11-04 (anomalyco/hex#1236,
  commit 8c4e9664), which keeps only the inode in new receipts.
