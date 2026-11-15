package yaml

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"
)

// Unmarshal decodes the document in data into the value v points to.
//
// What decodes into what:
//
//   - a mapping into a struct, by the fields' yaml tags (or their names in
//     lower case), or into a map with string keys;
//   - a list into a slice;
//   - a scalar into a string; into a bool if it is true or false; into an
//     integer or a float if it is a number, written without quotes; and into a
//     [time.Duration] if it is one as Go writes them: 90s, 1h30m;
//   - anything into an empty interface: maps, slices, strings, bools, ints
//     and floats, the way JSON would;
//   - null, or no value at all, into nothing: the field keeps what it had.
func Unmarshal(data []byte, v any) error {
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Pointer || rv.IsNil() {
		return errors.New("yaml: Unmarshal wants a non-nil pointer")
	}
	root, err := parse(data)
	if err != nil {
		return err
	}
	return decode(root, rv.Elem(), "")
}

var durationType = reflect.TypeFor[time.Duration]()

// decode stores n in v. path names v for an error message: server.addr,
// tools[2].
func decode(n *node, v reflect.Value, path string) error {
	if n.kind == nullKind {
		return nil
	}
	if v.Type() == durationType {
		s, err := scalar(n, path)
		if err != nil {
			return err
		}
		d, err := time.ParseDuration(s)
		if err != nil {
			return mismatch(n, path, "a duration, such as 90s or 1h30m")
		}
		v.SetInt(int64(d))
		return nil
	}

	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			v.Set(reflect.New(v.Type().Elem()))
		}
		return decode(n, v.Elem(), path)
	case reflect.Struct:
		return decodeStruct(n, v, path)
	case reflect.Map:
		return decodeMap(n, v, path)
	case reflect.Slice:
		return decodeSlice(n, v, path)
	case reflect.Interface:
		if v.NumMethod() > 0 {
			break
		}
		if g := generic(n); g != nil {
			v.Set(reflect.ValueOf(g))
		}
		return nil
	}

	s, err := scalar(n, path)
	if err != nil {
		return err
	}
	switch v.Kind() {
	case reflect.String:
		v.SetString(s)
		return nil
	case reflect.Bool:
		b, ok := boolean(n)
		if !ok {
			return mismatch(n, path, "true or false")
		}
		v.SetBool(b)
		return nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		i, err := integer(s, v.Type().Bits())
		if err != nil || !n.plain {
			return mismatch(n, path, "a whole number")
		}
		v.SetInt(i)
		return nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		i, err := integer(s, 64)
		if err != nil || i < 0 || !n.plain || v.OverflowUint(uint64(i)) {
			return mismatch(n, path, "a whole number that is not negative")
		}
		v.SetUint(uint64(i))
		return nil
	case reflect.Float32, reflect.Float64:
		f, err := strconv.ParseFloat(s, v.Type().Bits())
		if err != nil || !n.plain || !numeric(s) {
			return mismatch(n, path, "a number")
		}
		v.SetFloat(f)
		return nil
	}
	return &Error{Line: n.line, Msg: fmt.Sprintf("%s: cannot decode into a %s", name(path), v.Type())}
}

func decodeStruct(n *node, v reflect.Value, path string) error {
	if n.kind != mappingKind {
		return wrongKind(n, path, "a mapping of keys to values")
	}
	fields := fieldsOf(v.Type())
	for _, e := range n.entries {
		i := slices.IndexFunc(fields, func(f field) bool { return f.name == e.key })
		if i < 0 {
			names := make([]string, len(fields))
			for j, f := range fields {
				names[j] = f.name
			}
			where := " at the top level"
			if path != "" {
				where = " in " + path
			}
			return &Error{Line: e.line, Msg: fmt.Sprintf("unknown key %q%s (known: %s)",
				e.key, where, strings.Join(names, ", "))}
		}
		if err := decode(e.val, v.FieldByIndex(fields[i].index), join(path, e.key)); err != nil {
			return err
		}
	}
	return nil
}

type field struct {
	name  string
	index []int
}

// fieldsOf lists the fields of t a document may set, in declaration order.
func fieldsOf(t reflect.Type) []field {
	var out []field
	for _, f := range reflect.VisibleFields(t) {
		if !f.IsExported() || f.Anonymous {
			continue
		}
		name, _, _ := strings.Cut(f.Tag.Get("yaml"), ",")
		switch name {
		case "-":
			continue
		case "":
			name = strings.ToLower(f.Name)
		}
		out = append(out, field{name: name, index: f.Index})
	}
	return out
}

func decodeMap(n *node, v reflect.Value, path string) error {
	if n.kind != mappingKind {
		return wrongKind(n, path, "a mapping of keys to values")
	}
	t := v.Type()
	if t.Key().Kind() != reflect.String {
		return &Error{Line: n.line, Msg: fmt.Sprintf("%s: cannot decode into a %s", name(path), t)}
	}
	m := reflect.MakeMapWithSize(t, len(n.entries))
	for _, e := range n.entries {
		val := reflect.New(t.Elem()).Elem()
		if err := decode(e.val, val, join(path, e.key)); err != nil {
			return err
		}
		m.SetMapIndex(reflect.ValueOf(e.key).Convert(t.Key()), val)
	}
	v.Set(m)
	return nil
}

func decodeSlice(n *node, v reflect.Value, path string) error {
	if n.kind != sequenceKind {
		return wrongKind(n, path, "a list")
	}
	s := reflect.MakeSlice(v.Type(), len(n.items), len(n.items))
	for i, item := range n.items {
		if err := decode(item, s.Index(i), fmt.Sprintf("%s[%d]", path, i)); err != nil {
			return err
		}
	}
	v.Set(s)
	return nil
}

// generic is n as JSON would have it.
func generic(n *node) any {
	switch n.kind {
	case mappingKind:
		m := make(map[string]any, len(n.entries))
		for _, e := range n.entries {
			m[e.key] = generic(e.val)
		}
		return m
	case sequenceKind:
		s := make([]any, len(n.items))
		for i, item := range n.items {
			s[i] = generic(item)
		}
		return s
	case scalarKind:
		if b, ok := boolean(n); ok {
			return b
		}
		if n.plain && numeric(n.text) {
			if i, err := integer(n.text, 64); err == nil {
				return int(i)
			}
			if f, err := strconv.ParseFloat(n.text, 64); err == nil {
				return f
			}
		}
		return n.text
	}
	return nil
}

func scalar(n *node, path string) (string, error) {
	if n.kind != scalarKind {
		return "", wrongKind(n, path, "a single value")
	}
	return n.text, nil
}

// boolean reads the words YAML 1.2 means by true and false — not yes, no,
// on or off, which it reads as strings, and so does this.
func boolean(n *node) (bool, bool) {
	if !n.plain {
		return false, false
	}
	switch n.text {
	case "true", "True", "TRUE":
		return true, true
	case "false", "False", "FALSE":
		return false, true
	}
	return false, false
}

// integer reads a decimal number, or one written 0x, 0o or 0b. A leading
// zero does not make it octal: 012 is twelve, as in YAML 1.2.
func integer(s string, bits int) (int64, error) {
	base := 10
	if t := strings.TrimLeft(s, "+-"); len(t) > 1 && t[0] == '0' && strings.ContainsRune("xXoObB", rune(t[1])) {
		base = 0
	}
	return strconv.ParseInt(s, base, bits)
}

// numeric reports whether s looks like a number at all, so that a word such
// as "inf" or "nan" stays a word.
func numeric(s string) bool {
	t := strings.TrimLeft(s, "+-")
	return t != "" && (t[0] >= '0' && t[0] <= '9' || t[0] == '.' && len(t) > 1 && t[1] >= '0' && t[1] <= '9')
}

func mismatch(n *node, path, want string) error {
	if !n.plain {
		return &Error{Line: n.line, Msg: fmt.Sprintf("%s: %q is quoted, which makes it a string; want %s",
			name(path), n.text, want)}
	}
	return &Error{Line: n.line, Msg: fmt.Sprintf("%s: %q is not %s", name(path), n.text, want)}
}

func wrongKind(n *node, path, want string) error {
	got := map[kind]string{scalarKind: "a single value", mappingKind: "a mapping", sequenceKind: "a list"}[n.kind]
	return &Error{Line: n.line, Msg: fmt.Sprintf("%s wants %s, not %s", name(path), want, got)}
}

func join(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

// name is path for a message, which needs something to call the top level.
func name(path string) string {
	if path == "" {
		return "the document"
	}
	return path
}
