package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

const stateMagic = "MDRFSV1\x00"
const stateSize = 8 + 30 + 30 + 20 + 20

var faceOrder = []string{"U", "R", "F", "L", "BR", "BL", "FR", "FL", "DR", "DL", "B", "D"}

type strictMove struct {
	face  string
	power int
}

func permutationParity(values []uint8) (int, error) {
	seen := make([]bool, len(values))
	for _, value := range values {
		if int(value) >= len(values) || seen[value] {
			return 0, errors.New("piece array is not a permutation")
		}
		seen[value] = true
	}
	parity := 0
	for i := 0; i < len(values); i++ {
		for j := i + 1; j < len(values); j++ {
			if values[i] > values[j] {
				parity ^= 1
			}
		}
	}
	return parity, nil
}

func validateState(state totalState) error {
	edges := make([]uint8, 30)
	corners := make([]uint8, 20)
	edgeOrientationSum := 0
	cornerOrientationSum := 0
	for position := 0; position < 30; position++ {
		edges[position] = state.edgePositions[position].piece
		if state.edgePositions[position].orientation {
			edgeOrientationSum++
		}
	}
	for position := 0; position < 20; position++ {
		corners[position] = state.cornerPositions[position].piece
		if state.cornerPositions[position].orientation > 2 {
			return errors.New("corner orientation is outside 0..2")
		}
		cornerOrientationSum += int(state.cornerPositions[position].orientation)
	}
	edgeParity, err := permutationParity(edges)
	if err != nil {
		return fmt.Errorf("edges: %w", err)
	}
	cornerParity, err := permutationParity(corners)
	if err != nil {
		return fmt.Errorf("corners: %w", err)
	}
	if edgeOrientationSum%2 != 0 {
		return errors.New("edge orientation sum is odd")
	}
	if cornerOrientationSum%3 != 0 {
		return errors.New("corner orientation sum is nonzero modulo 3")
	}
	if edgeParity != 0 || cornerParity != 0 {
		return errors.New("Megaminx edge and corner permutations must both be even")
	}
	return nil
}

func encodeState(state totalState) ([]byte, error) {
	if err := validateState(state); err != nil {
		return nil, err
	}
	result := make([]byte, stateSize)
	copy(result[:8], []byte(stateMagic))
	offset := 8
	for position := 0; position < 30; position++ {
		result[offset+position] = state.edgePositions[position].piece
	}
	offset += 30
	for position := 0; position < 30; position++ {
		if state.edgePositions[position].orientation {
			result[offset+position] = 1
		}
	}
	offset += 30
	for position := 0; position < 20; position++ {
		result[offset+position] = state.cornerPositions[position].piece
	}
	offset += 20
	for position := 0; position < 20; position++ {
		result[offset+position] = state.cornerPositions[position].orientation
	}
	return result, nil
}

func decodeState(data []byte) (totalState, error) {
	if len(data) != stateSize {
		return totalState{}, fmt.Errorf("FullStateV1 must be %d bytes, got %d", stateSize, len(data))
	}
	if !bytes.Equal(data[:8], []byte(stateMagic)) {
		return totalState{}, errors.New("invalid FullStateV1 magic")
	}
	state := totalState{}
	offset := 8
	for position := 0; position < 30; position++ {
		piece := data[offset+position]
		state.edgePositions[position] = edgePosition{piece: piece, orientation: data[offset+30+position] == 1}
		if data[offset+30+position] > 1 || piece >= 30 {
			return totalState{}, errors.New("invalid edge record")
		}
		state.edgePieces[piece] = uint8(position)
	}
	offset += 60
	for position := 0; position < 20; position++ {
		piece := data[offset+position]
		orientation := data[offset+20+position]
		if piece >= 20 || orientation > 2 {
			return totalState{}, errors.New("invalid corner record")
		}
		state.cornerPositions[position] = cornerPosition{piece: piece, orientation: orientation}
		state.cornerPieces[piece] = uint8(position)
	}
	if err := validateState(state); err != nil {
		return totalState{}, err
	}
	return state, nil
}

func parseStrictWord(data []byte) ([]strictMove, error) {
	if bytes.Contains(data, []byte{'\r'}) {
		return nil, errors.New("CR is forbidden in canonical move words")
	}
	if len(data) > 0 && data[len(data)-1] == '\n' {
		data = data[:len(data)-1]
	}
	if bytes.Contains(data, []byte{'\n'}) {
		return nil, errors.New("move word must occupy one line")
	}
	if len(data) == 0 {
		return []strictMove{}, nil
	}
	text := string(data)
	if strings.HasPrefix(text, " ") || strings.HasSuffix(text, " ") || strings.Contains(text, "  ") {
		return nil, errors.New("tokens must be separated by one ASCII space")
	}
	validFaces := map[string]bool{}
	for _, face := range faceOrder {
		validFaces[face] = true
	}
	parts := strings.Split(text, " ")
	result := make([]strictMove, 0, len(parts))
	for _, token := range parts {
		if len(token) < 2 {
			return nil, fmt.Errorf("invalid move token %q", token)
		}
		powerByte := token[len(token)-1]
		face := token[:len(token)-1]
		if !validFaces[face] || powerByte < '1' || powerByte > '4' {
			return nil, fmt.Errorf("invalid move token %q", token)
		}
		result = append(result, strictMove{face: face, power: int(powerByte - '0')})
	}
	return result, nil
}

func applyBaseMove(state totalState, face string) (totalState, error) {
	switch face {
	case "U": return moveU(state), nil
	case "R": return moveR(state), nil
	case "F": return moveF(state), nil
	case "L": return moveL(state), nil
	case "BR": return moveBR(state), nil
	case "BL": return moveBL(state), nil
	case "FR": return moveFR(state), nil
	case "FL": return moveFL(state), nil
	case "DR": return moveDR(state), nil
	case "DL": return moveDL(state), nil
	case "B": return moveB(state), nil
	case "D": return moveD(state), nil
	default: return totalState{}, fmt.Errorf("unknown face %q", face)
	}
}

func applyWord(state totalState, word []strictMove) (totalState, error) {
	var err error
	for _, current := range word {
		for power := 0; power < current.power; power++ {
			state, err = applyBaseMove(state, current.face)
			if err != nil { return totalState{}, err }
		}
	}
	return state, validateState(state)
}

func equalState(left, right totalState) bool {
	leftBytes, leftErr := encodeState(left)
	rightBytes, rightErr := encodeState(right)
	return leftErr == nil && rightErr == nil && bytes.Equal(leftBytes, rightBytes)
}

func subgroupFaces(name string) ([]string, error) {
	counts := map[string]int{"g5": 7, "g6": 6, "g7": 5, "g8": 4, "g9": 3}
	count, ok := counts[strings.ToLower(name)]
	if !ok { return nil, fmt.Errorf("unknown target %q", name) }
	return faceOrder[:count], nil
}

func inTarget(state totalState, target string) (bool, error) {
	if strings.ToLower(target) == "solved" {
		return equalState(state, defaultState), nil
	}
	faces, err := subgroupFaces(target)
	if err != nil { return false, err }
	mobileEdges := [30]bool{}
	mobileCorners := [20]bool{}
	for _, face := range faces {
		moved, _ := applyBaseMove(defaultState, face)
		for i := 0; i < 30; i++ {
			if moved.edgePositions[i] != defaultState.edgePositions[i] { mobileEdges[i] = true }
		}
		for i := 0; i < 20; i++ {
			if moved.cornerPositions[i] != defaultState.cornerPositions[i] { mobileCorners[i] = true }
		}
	}
	for i := 0; i < 30; i++ {
		if !mobileEdges[i] && state.edgePositions[i] != defaultState.edgePositions[i] { return false, nil }
	}
	for i := 0; i < 20; i++ {
		if !mobileCorners[i] && state.cornerPositions[i] != defaultState.cornerPositions[i] { return false, nil }
	}
	return true, nil
}

func readWord(path string) ([]strictMove, error) {
	data, err := os.ReadFile(path)
	if err != nil { return nil, err }
	return parseStrictWord(data)
}

func readState(path string) (totalState, error) {
	data, err := os.ReadFile(path)
	if err != nil { return totalState{}, err }
	return decodeState(data)
}

func writeState(path string, state totalState) error {
	data, err := encodeState(state)
	if err != nil { return err }
	temporary := path + ".partial"
	if err := os.WriteFile(temporary, data, 0644); err != nil { return err }
	return os.Rename(temporary, path)
}

func runMakeState(arguments []string) error {
	flags := flag.NewFlagSet("make-state", flag.ContinueOnError)
	movesPath := flags.String("moves", "", "canonical move word")
	outPath := flags.String("out", "", "FullStateV1 output")
	if err := flags.Parse(arguments); err != nil { return err }
	if *movesPath == "" || *outPath == "" || flags.NArg() != 0 { return errors.New("make-state requires --moves and --out") }
	word, err := readWord(*movesPath); if err != nil { return err }
	state, err := applyWord(defaultState, word); if err != nil { return err }
	return writeState(*outPath, state)
}

func runApply(arguments []string) error {
	flags := flag.NewFlagSet("apply", flag.ContinueOnError)
	statePath := flags.String("state", "", "FullStateV1 input")
	movesPath := flags.String("moves", "", "canonical move word")
	outPath := flags.String("out", "", "FullStateV1 output")
	if err := flags.Parse(arguments); err != nil { return err }
	if *statePath == "" || *movesPath == "" || *outPath == "" || flags.NArg() != 0 { return errors.New("apply requires --state, --moves, and --out") }
	state, err := readState(*statePath); if err != nil { return err }
	word, err := readWord(*movesPath); if err != nil { return err }
	state, err = applyWord(state, word); if err != nil { return err }
	return writeState(*outPath, state)
}

func runVerify(arguments []string) error {
	flags := flag.NewFlagSet("verify", flag.ContinueOnError)
	statePath := flags.String("state", "", "FullStateV1 input")
	solutionPath := flags.String("solution", "", "canonical move word")
	target := flags.String("target", "solved", "solved or g5..g9")
	maxLength := flags.Int("max-length", -1, "maximum FTM length")
	if err := flags.Parse(arguments); err != nil { return err }
	if *statePath == "" || *solutionPath == "" || *maxLength < 0 || flags.NArg() != 0 { return errors.New("verify requires --state, --solution, and nonnegative --max-length") }
	state, err := readState(*statePath); if err != nil { return err }
	word, err := readWord(*solutionPath); if err != nil { return err }
	if len(word) > *maxLength { return fmt.Errorf("solution length %d exceeds bound %d", len(word), *maxLength) }
	state, err = applyWord(state, word); if err != nil { return err }
	ok, err := inTarget(state, *target); if err != nil { return err }
	if !ok { return fmt.Errorf("solution does not reach target %s", *target) }
	output := map[string]any{"valid": true, "length": len(word), "target": strings.ToLower(*target)}
	encoded, _ := json.Marshal(output)
	fmt.Println(string(encoded))
	return nil
}

func runVerifyBatch(arguments []string) error {
	flags := flag.NewFlagSet("verify-batch", flag.ContinueOnError)
	target := flags.String("target", "solved", "solved or g5..g9")
	maxLength := flags.Int("max-length", -1, "maximum FTM length")
	if err := flags.Parse(arguments); err != nil { return err }
	if *maxLength < 0 || flags.NArg() != 0 {
		return errors.New("verify-batch requires a nonnegative --max-length and no positional arguments")
	}
	normalizedTarget := strings.ToLower(*target)
	if normalizedTarget != "solved" {
		if _, err := subgroupFaces(normalizedTarget); err != nil { return err }
	}

	// Each stdin line is: decimal state ID, tab, FullStateV1 hex, tab,
	// canonical move word.  IDs must be strictly increasing, allowing the
	// Python database checker and this independent replay to agree on an exact
	// record count without giving the Go implementation SQLite dependencies.
	scanner := bufio.NewScanner(os.Stdin)
	// Unlike bufio.ScanLines, this splitter preserves carriage returns and
	// rejects an unterminated final record.  The accepted transcript is thus a
	// canonical byte sequence: every record ends in exactly one LF byte.
	scanner.Split(func(data []byte, atEOF bool) (advance int, token []byte, err error) {
		if index := bytes.IndexByte(data, '\n'); index >= 0 {
			return index + 1, data[:index], nil
		}
		if atEOF && len(data) != 0 {
			return 0, nil, errors.New("unterminated final batch record")
		}
		return 0, nil, nil
	})
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	records := 0
	maximum := 0
	var previousID uint64
	var firstID uint64
	havePrevious := false
	transcript := sha256.New()
	for scanner.Scan() {
		line := scanner.Bytes()
		fields := strings.Split(string(line), "\t")
		if len(fields) != 3 {
			return fmt.Errorf("batch record %d must have three tab-separated fields", records+1)
		}
		stateID, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil { return fmt.Errorf("batch record %d has invalid state ID: %w", records+1, err) }
		if havePrevious && stateID <= previousID {
			return fmt.Errorf("batch state IDs are not strictly increasing at %d", stateID)
		}
		stateData, err := hex.DecodeString(fields[1])
		if err != nil { return fmt.Errorf("batch state %d has invalid hex: %w", stateID, err) }
		state, err := decodeState(stateData)
		if err != nil { return fmt.Errorf("batch state %d is invalid: %w", stateID, err) }
		word, err := parseStrictWord([]byte(fields[2]))
		if err != nil { return fmt.Errorf("batch state %d has invalid word: %w", stateID, err) }
		if len(word) > *maxLength {
			return fmt.Errorf("batch state %d solution length %d exceeds bound %d", stateID, len(word), *maxLength)
		}
		state, err = applyWord(state, word)
		if err != nil { return fmt.Errorf("batch state %d replay failed: %w", stateID, err) }
		ok, err := inTarget(state, normalizedTarget)
		if err != nil { return fmt.Errorf("batch state %d target check failed: %w", stateID, err) }
		if !ok { return fmt.Errorf("batch state %d does not reach target %s", stateID, normalizedTarget) }
		if len(word) > maximum { maximum = len(word) }
		if !havePrevious { firstID = stateID }
		previousID = stateID
		havePrevious = true
		_, _ = transcript.Write(line)
		_, _ = transcript.Write([]byte{'\n'})
		records++
	}
	if err := scanner.Err(); err != nil { return fmt.Errorf("cannot read batch stream: %w", err) }
	var maximumOutput any
	var firstIDOutput any
	var lastIDOutput any
	if records > 0 { maximumOutput = maximum }
	if records > 0 { firstIDOutput = firstID; lastIDOutput = previousID }
	output := map[string]any{
		"valid": true,
		"records": records,
		"maximum_solution_length": maximumOutput,
		"target": normalizedTarget,
		"max_length": *maxLength,
		"first_state_id": firstIDOutput,
		"last_state_id": lastIDOutput,
		"transcript_sha256": fmt.Sprintf("%x", transcript.Sum(nil)),
	}
	encoded, _ := json.Marshal(output)
	fmt.Println(string(encoded))
	return nil
}

func runInspect(arguments []string) error {
	flags := flag.NewFlagSet("inspect", flag.ContinueOnError)
	statePath := flags.String("state", "", "FullStateV1 input")
	if err := flags.Parse(arguments); err != nil { return err }
	if *statePath == "" || flags.NArg() != 0 { return errors.New("inspect requires --state") }
	state, err := readState(*statePath); if err != nil { return err }
	targets := []string{"solved", "g5", "g6", "g7", "g8", "g9"}
	result := map[string]any{"format": "FullStateV1", "bytes": stateSize, "targets": map[string]bool{}}
	for _, target := range targets { ok, _ := inTarget(state, target); result["targets"].(map[string]bool)[target] = ok }
	encoded, _ := json.Marshal(result); fmt.Println(string(encoded))
	return nil
}

func runPhaseIndex(arguments []string) error {
	flags := flag.NewFlagSet("phase-index", flag.ContinueOnError)
	statePath := flags.String("state", "", "FullStateV1 input")
	phase := flags.Int("phase", 0, "phase number 3..6")
	if err := flags.Parse(arguments); err != nil { return err }
	if *statePath == "" || *phase < 3 || *phase > 6 || flags.NArg() != 0 {
		return errors.New("phase-index requires --state and --phase 3..6")
	}
	state, err := readState(*statePath); if err != nil { return err }
	var index, solvedIndex uint32
	switch *phase {
	case 3: index, solvedIndex = hash7Gen(state), hash7Gen(defaultState)
	case 4: index, solvedIndex = hash6Gen(state), hash6Gen(defaultState)
	case 5: index, solvedIndex = hash5Gen(state), hash5Gen(defaultState)
	case 6: index, solvedIndex = hash4Gen(state), hash4Gen(defaultState)
	}
	encoded, _ := json.Marshal(map[string]any{
		"phase": *phase, "index": index, "solved_index": solvedIndex,
	})
	fmt.Println(string(encoded))
	return nil
}

func runExport(arguments []string) error {
	if len(arguments) != 0 { return errors.New("export-moves takes no arguments") }
	result := map[string]string{}
	for _, face := range faceOrder {
		state, _ := applyBaseMove(defaultState, face)
		encoded, _ := encodeState(state)
		result[face+"1"] = fmt.Sprintf("%x", encoded)
	}
	keys := make([]string, 0, len(result)); for key := range result { keys = append(keys, key) }; sort.Strings(keys)
	ordered := make([]map[string]string, 0, len(keys)); for _, key := range keys { ordered = append(ordered, map[string]string{"move": key, "state_hex": result[key]}) }
	encoded, _ := json.Marshal(ordered); fmt.Println(string(encoded))
	return nil
}

func main() {
	if len(os.Args) < 2 { fmt.Fprintln(os.Stderr, "usage: mdr-verify <make-state|apply|verify|verify-batch|inspect|phase-index|export-moves>"); os.Exit(2) }
	var err error
	switch os.Args[1] {
	case "make-state": err = runMakeState(os.Args[2:])
	case "apply": err = runApply(os.Args[2:])
	case "verify": err = runVerify(os.Args[2:])
	case "verify-batch": err = runVerifyBatch(os.Args[2:])
	case "inspect": err = runInspect(os.Args[2:])
	case "phase-index": err = runPhaseIndex(os.Args[2:])
	case "export-moves": err = runExport(os.Args[2:])
	default: err = fmt.Errorf("unknown command %q", os.Args[1])
	}
	if err != nil { fmt.Fprintln(os.Stderr, "error:", err); os.Exit(2) }
}
