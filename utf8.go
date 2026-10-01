package lfm

// completeUTF8 returns how many leading bytes of b can be shown now: all of b, except a trailing
// lead byte (and its continuation bytes) of a multi-byte character that still misses bytes. A
// byte-level token can end in the middle of a character, and showing half of it prints garbage.
func completeUTF8(b []byte) int {
	for back := 1; back <= 3 && back <= len(b); back++ {
		c := b[len(b)-back]
		if c&0xC0 == 0x80 { // continuation byte: the lead is further back
			continue
		}
		need := 1
		switch {
		case c&0xE0 == 0xC0:
			need = 2
		case c&0xF0 == 0xE0:
			need = 3
		case c&0xF8 == 0xF0:
			need = 4
		}
		if need > back {
			return len(b) - back
		}
		return len(b)
	}
	return len(b)
}
