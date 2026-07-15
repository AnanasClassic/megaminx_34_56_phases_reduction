package main

import (
	"crypto/sha256"
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

const upstreamCommit = "82db40b5744297617446da888cd73d5e26f57239"
const unseen = byte(255)

var faces = []string{"U", "R", "F", "L", "BR", "BL", "FR", "FL", "DR", "DL", "B", "D"}

type moveSpec struct {
	name string
	r    bool
	c    [5]uint8
	e    [5]uint8
}

var specs = []moveSpec{
	{"U", false, [5]uint8{0, 1, 2, 3, 4}, [5]uint8{0, 1, 2, 3, 4}},
	{"R", true, [5]uint8{0, 4, 9, 11, 5}, [5]uint8{4, 9, 14, 16, 5}},
	{"F", true, [5]uint8{1, 0, 5, 10, 6}, [5]uint8{0, 5, 10, 15, 6}},
	{"L", true, [5]uint8{2, 1, 6, 14, 7}, [5]uint8{1, 6, 11, 19, 7}},
	{"BR", true, [5]uint8{4, 3, 8, 12, 9}, [5]uint8{3, 8, 13, 17, 9}},
	{"BL", true, [5]uint8{3, 2, 7, 13, 8}, [5]uint8{2, 7, 12, 18, 8}},
	{"FR", true, [5]uint8{16, 15, 10, 5, 11}, [5]uint8{25, 20, 10, 16, 21}},
	{"FL", true, [5]uint8{15, 19, 14, 6, 10}, [5]uint8{29, 24, 11, 15, 20}},
	{"DR", true, [5]uint8{17, 16, 11, 9, 12}, [5]uint8{26, 21, 14, 17, 22}},
	{"DL", true, [5]uint8{19, 18, 13, 7, 14}, [5]uint8{28, 23, 12, 19, 24}},
	{"B", true, [5]uint8{18, 17, 12, 8, 13}, [5]uint8{27, 22, 13, 18, 23}},
	{"D", false, [5]uint8{15, 16, 17, 18, 19}, [5]uint8{25, 26, 27, 28, 29}},
}

type tracked struct { p [3]uint8; o [3]uint8 }

type layout struct {
	phase, faceCount int
	transition string
	stateCount, expectedDiameter, expectedAntipodes int
	cornerAllowed, edgeAllowed []uint8
	cornerPieces [2]uint8
	edgePieces [3]uint8
	cornerCount, edgeCount uint32
	cornerRep, edgeRep []tracked
	cornerTransitions, edgeTransitions [][]uint32
	cornerOrdinal, edgeOrdinal [30]int
}

func phaseLayout(phase int) (*layout, error) {
	var l layout
	switch phase {
	case 3:
		l = layout{phase:3, faceCount:7, transition:"G5->G6", stateCount:208099584, expectedDiameter:14, expectedAntipodes:212,
			cornerAllowed: seq(0, 16), edgeAllowed: append(seq(0, 21), 25), cornerPieces:[2]uint8{15,16}, edgePieces:[3]uint8{20,21,25}}
	case 5:
		l = layout{phase:5, faceCount:5, transition:"G7->G8", stateCount:64157184, expectedDiameter:13, expectedAntipodes:3484,
			cornerAllowed: []uint8{0,1,2,3,4,5,6,7,8,9,10,11,12,14},
			edgeAllowed: []uint8{0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,19}, cornerPieces:[2]uint8{8,12}, edgePieces:[3]uint8{8,13,17}}
	case 6:
		l = layout{phase:6, faceCount:4, transition:"G8->G9", stateCount:25945920, expectedDiameter:13, expectedAntipodes:117,
			cornerAllowed: []uint8{0,1,2,3,4,5,6,7,9,10,11,14},
			edgeAllowed: []uint8{0,1,2,3,4,5,6,7,9,10,11,14,15,16,19}, cornerPieces:[2]uint8{7,14}, edgePieces:[3]uint8{7,11,19}}
	default: return nil, fmt.Errorf("phase %d is not a dense coordinate", phase)
	}
	l.initialize()
	return &l, nil
}

func seq(first, last int) []uint8 { r:=make([]uint8,last-first+1); for i:=range r { r[i]=uint8(first+i) }; return r }

func (l *layout) initialize() {
	for i:=range l.cornerOrdinal { l.cornerOrdinal[i]=-1; l.edgeOrdinal[i]=-1 }
	for i,p:=range l.cornerAllowed { l.cornerOrdinal[p]=i }
	for i,p:=range l.edgeAllowed { l.edgeOrdinal[p]=i }
	nc, ne := len(l.cornerAllowed), len(l.edgeAllowed)
	l.cornerCount = uint32(nc*(nc-1)*9)
	l.edgeCount = uint32(ne*(ne-1)*(ne-2)*8)
	if int(l.cornerCount*l.edgeCount) != l.stateCount { panic("coordinate product does not match state count") }
	l.cornerRep = make([]tracked,l.cornerCount); cornerSeen:=make([]bool,l.cornerCount)
	for _,a:=range l.cornerAllowed { for _,b:=range l.cornerAllowed { if a==b {continue}; for o:=0;o<9;o++ {
		t:=tracked{p:[3]uint8{a,b},o:[3]uint8{uint8(o%3),uint8(o/3)}}; i:=l.rankCorner(t); l.cornerRep[i]=t; cornerSeen[i]=true
	} } }
	l.edgeRep = make([]tracked,l.edgeCount); edgeSeen:=make([]bool,l.edgeCount)
	for _,a:=range l.edgeAllowed { for _,b:=range l.edgeAllowed { if a==b {continue}; for _,c:=range l.edgeAllowed { if c==a||c==b {continue}; for o:=0;o<8;o++ {
		t:=tracked{p:[3]uint8{a,b,c},o:[3]uint8{uint8((o>>2)&1),uint8((o>>1)&1),uint8(o&1)}}; i:=l.rankEdge(t); l.edgeRep[i]=t; edgeSeen[i]=true
	} } } }
	for _,v:=range cornerSeen { if !v {panic("corner rank is not bijective")} }; for _,v:=range edgeSeen {if !v {panic("edge rank is not bijective")}}
	moves:=l.faceCount*4; l.cornerTransitions=make([][]uint32,moves); l.edgeTransitions=make([][]uint32,moves)
	for code:=0;code<moves;code++ { l.cornerTransitions[code]=make([]uint32,l.cornerCount); l.edgeTransitions[code]=make([]uint32,l.edgeCount); spec:=specs[code/4]; power:=code%4+1
		for i,t:=range l.cornerRep { for k:=0;k<power;k++ { t=applyTracked(t,spec,true,2) }; l.cornerTransitions[code][i]=l.rankCorner(t) }
		for i,t:=range l.edgeRep { for k:=0;k<power;k++ { t=applyTracked(t,spec,false,3) }; l.edgeTransitions[code][i]=l.rankEdge(t) }
	}
}

func orderedRank(values []int, n int) int {
	r:=0; scale:=1
	for i,v:=range values { q:=v; for j:=0;j<i;j++ {if values[j]<v {q--}}; r+=q*scale; scale*=n-i }
	return r
}
func (l *layout) rankCorner(t tracked) uint32 { a:=l.cornerOrdinal[t.p[0]]; b:=l.cornerOrdinal[t.p[1]]; if a<0||b<0||a==b {panic("invalid corner position")}; return uint32(int(t.o[0])+3*int(t.o[1])+9*orderedRank([]int{a,b},len(l.cornerAllowed))) }
func (l *layout) rankEdge(t tracked) uint32 { a:=l.edgeOrdinal[t.p[0]]; b:=l.edgeOrdinal[t.p[1]]; c:=l.edgeOrdinal[t.p[2]]; if a<0||b<0||c<0||a==b||a==c||b==c {panic("invalid edge position")}; o:=int(t.o[0])*4+int(t.o[1])*2+int(t.o[2]); return uint32(o+8*orderedRank([]int{a,b,c},len(l.edgeAllowed))) }

func applyTracked(t tracked, s moveSpec, corner bool, count int) tracked {
	cycle:=s.e; if corner {cycle=s.c}
	for piece:=0;piece<count;piece++ { for i,p:=range cycle { if t.p[piece]!=p {continue}; if s.r {if corner {if i==0 {t.o[piece]=(t.o[piece]+1)%3} else {t.o[piece]=(t.o[piece]+2)%3}} else if i==2||i==4 {t.o[piece]^=1}}; t.p[piece]=cycle[(i+1)%5]; break } }
	return t
}

func (l *layout) denseFromState(s totalState) uint32 {
	c:=tracked{}; e:=tracked{}
	for i,piece:=range l.cornerPieces {p:=s.cornerPieces[piece]; c.p[i]=p; c.o[i]=s.cornerPositions[p].orientation}
	for i,piece:=range l.edgePieces {p:=s.edgePieces[piece]; e.p[i]=p; if s.edgePositions[p].orientation {e.o[i]=1}}
	return l.rankCorner(c)+l.cornerCount*l.rankEdge(e)
}

func (l *layout) upstreamFromDense(index uint32) uint32 { cr:=index%l.cornerCount; er:=index/l.cornerCount; return (cr+l.cornerCount*(er/8))*8+er%8 }
func (l *layout) next(index uint32, code int) uint32 { return l.cornerTransitions[code][index%l.cornerCount]+l.cornerCount*l.edgeTransitions[code][index/l.cornerCount] }
func inverseCode(code int) byte { return byte((code/4)*4+(4-(code%4+1))) }

func applyFull(s totalState, code int) totalState { for i:=0;i<code%4+1;i++ {switch code/4 {case 0:s=moveU(s);case 1:s=moveR(s);case 2:s=moveF(s);case 3:s=moveL(s);case 4:s=moveBR(s);case 5:s=moveBL(s);case 6:s=moveFR(s)}}; return s }
func upstreamHash(phase int,s totalState) uint32 {switch phase {case 3:return hash7Gen(s);case 4:return hash6Gen(s);case 5:return hash5Gen(s);case 6:return hash4Gen(s)};panic("phase")}

func selfTest() error {
	for _,phase:=range []int{3,5,6} { l,_:=phaseLayout(phase)
		for face:=0;face<l.faceCount;face++ { code:=face*4; for i:=uint32(0);i<l.cornerCount;i++ {x:=i;for k:=0;k<5;k++ {x=l.cornerTransitions[code][x]};if x!=i{return fmt.Errorf("phase %d corner order face %d",phase,face)}}; for i:=uint32(0);i<l.edgeCount;i++ {x:=i;for k:=0;k<5;k++ {x=l.edgeTransitions[code][x]};if x!=i{return fmt.Errorf("phase %d edge order face %d",phase,face)}} }
		s:=defaultState
		for step:=0;step<500;step++ { code:=(step*17+11)%(l.faceCount*4); dense:=l.denseFromState(s); if l.upstreamFromDense(dense)!=upstreamHash(phase,s) {return fmt.Errorf("phase %d rank mismatch at step %d",phase,step)}; moved:=applyFull(s,code); if l.next(dense,code)!=l.denseFromState(moved) {return fmt.Errorf("phase %d transition mismatch at step %d",phase,step)}; s=moved }
		fmt.Printf("phase %d: exhaustive component bijections/order and 500 upstream transitions OK\n",phase)
	}
	return nil
}

type checkpoint struct { SchemaVersion int `json:"schema_version"`; Phase, Layer, StorageSlots, FrontierCount int; Complete bool `json:"complete"` }

func atomicWrite(path string,data []byte) error { if err:=os.MkdirAll(filepath.Dir(path),0755);err!=nil{return err}; tmp:=path+".partial"; f,err:=os.Create(tmp);if err!=nil{return err};if _,err=f.Write(data);err==nil{err=f.Sync()};cerr:=f.Close();if err==nil{err=cerr};if err!=nil{return err};return os.Rename(tmp,path) }
func atomicJSON(path string,v any) error {b,err:=json.MarshalIndent(v,"","  ");if err!=nil{return err};b=append(b,'\n');return atomicWrite(path,b)}
func encodeFrontier(v []uint32) []byte {b:=make([]byte,4*len(v));for i,x:=range v{binary.LittleEndian.PutUint32(b[4*i:],x)};return b}
func decodeFrontier(b []byte)([]uint32,error){if len(b)%4!=0{return nil,errors.New("bad frontier")};v:=make([]uint32,len(b)/4);for i:=range v{v[i]=binary.LittleEndian.Uint32(b[4*i:])};return v,nil}

func saveCheckpoint(work string,phase,layer int,depth,parent []byte,frontier []uint32) error { if err:=atomicWrite(filepath.Join(work,"depths.bin"),depth);err!=nil{return err};if err:=atomicWrite(filepath.Join(work,"predecessors.bin"),parent);err!=nil{return err};if err:=atomicWrite(filepath.Join(work,"frontier.bin"),encodeFrontier(frontier));err!=nil{return err};return atomicJSON(filepath.Join(work,"checkpoint.json"),checkpoint{1,phase,layer,len(depth),len(frontier),false}) }
func loadCheckpoint(work string,phase,slots int)([]byte,[]byte,[]uint32,int,error){b,err:=os.ReadFile(filepath.Join(work,"checkpoint.json"));if err!=nil{return nil,nil,nil,0,err};var c checkpoint;if json.Unmarshal(b,&c)!=nil||c.SchemaVersion!=1||c.Phase!=phase||c.StorageSlots!=slots{return nil,nil,nil,0,errors.New("incompatible checkpoint")};d,err:=os.ReadFile(filepath.Join(work,"depths.bin"));if err!=nil{return nil,nil,nil,0,err};p,err:=os.ReadFile(filepath.Join(work,"predecessors.bin"));if err!=nil{return nil,nil,nil,0,err};f,err:=os.ReadFile(filepath.Join(work,"frontier.bin"));if err!=nil{return nil,nil,nil,0,err};frontier,err:=decodeFrontier(f);if err!=nil||len(d)!=slots||len(p)!=slots||len(frontier)!=c.FrontierCount{return nil,nil,nil,0,errors.New("corrupt checkpoint")};return d,p,frontier,c.Layer,nil}

func buildDense(l *layout,out string,workers int,resume bool) ([]byte,[]byte,error) {
	work:=filepath.Join(out,".work"); slots:=l.stateCount; var depth,parent []byte;var frontier []uint32;layer:=0
	if resume {var err error;depth,parent,frontier,layer,err=loadCheckpoint(work,l.phase,slots);if err!=nil{return nil,nil,err};fmt.Printf("resumed phase %d layer %d frontier %d\n",l.phase,layer,len(frontier))} else {depth=make([]byte,slots);parent=make([]byte,slots);for i:=range depth{depth[i]=unseen;parent[i]=unseen};solved:=l.denseFromState(defaultState);depth[solved]=0;frontier=[]uint32{solved};if err:=saveCheckpoint(work,l.phase,0,depth,parent,frontier);err!=nil{return nil,nil,err}}
	candidates:=make([]atomic.Uint32,slots)
	for len(frontier)>0 { tag:=uint32(layer+1);locals:=make([][]uint32,workers);var cursor atomic.Uint64;var wg sync.WaitGroup
		for worker:=0;worker<workers;worker++ {wg.Add(1);go func(id int){defer wg.Done();local:=make([]uint32,0,len(frontier)/workers);for {start:=int(cursor.Add(1024)-1024);if start>=len(frontier){break};end:=start+1024;if end>len(frontier){end=len(frontier)};for _,index:=range frontier[start:end]{for code:=0;code<l.faceCount*4;code++{next:=l.next(index,code);if depth[next]!=unseen{continue};value:=(tag<<8)|uint32(inverseCode(code)+1);cell:=&candidates[next];for{old:=cell.Load();if old>>8==tag&&old&255<=value&255{break};if cell.CompareAndSwap(old,value){if old>>8!=tag{local=append(local,next)};break}}}}};locals[id]=local}(worker)}
		wg.Wait();total:=0;for _,v:=range locals{total+=len(v)};nextFrontier:=make([]uint32,0,total);for _,v:=range locals{nextFrontier=append(nextFrontier,v...)};sort.Slice(nextFrontier,func(i,j int)bool{return nextFrontier[i]<nextFrontier[j]});for _,index:=range nextFrontier{value:=candidates[index].Load();depth[index]=byte(layer+1);parent[index]=byte(value&255-1)};layer++;fmt.Printf("phase %d depth %d: %d\n",l.phase,layer,len(nextFrontier));frontier=nextFrontier;if err:=saveCheckpoint(work,l.phase,layer,depth,parent,frontier);err!=nil{return nil,nil,err}
	}
	for index,d:=range depth { if d==unseen||d==0 {continue}; code:=int(parent[index]); if code<0||code>=l.faceCount*4{return nil,nil,fmt.Errorf("invalid predecessor move at %d",index)}; if depth[l.next(uint32(index),code)]+1!=d{return nil,nil,fmt.Errorf("predecessor does not lower index %d",index)} }
	return depth,parent,nil
}

func buildSparse4(out string)([]byte,[]byte,[]uint32,error){slots:=103224;depth:=make([]byte,slots);parent:=make([]byte,slots);pred:=make([]uint32,slots);for i:=range depth{depth[i]=unseen;parent[i]=unseen;pred[i]=^uint32(0)};root:=hash6Gen(defaultState);depth[root]=0;frontier:=map[uint32]totalState{root:defaultState};layer:=0
	for len(frontier)>0 {keys:=make([]int,0,len(frontier));for k:=range frontier{keys=append(keys,int(k))};sort.Ints(keys);next:=map[uint32]totalState{};for _,ki:=range keys{s:=frontier[uint32(ki)];for code:=0;code<24;code++{m:=applyFull(s,code);h:=hash6Gen(m);if depth[h]==unseen{depth[h]=byte(layer+1);parent[h]=inverseCode(code);pred[h]=uint32(ki);next[h]=m}}};layer++;fmt.Printf("phase 4 depth %d: %d\n",layer,len(next));frontier=next};for i,d:=range depth{if d==unseen||d==0{continue};if pred[i]==^uint32(0)||depth[pred[i]]+1!=d{return nil,nil,nil,fmt.Errorf("invalid phase 4 predecessor at %d",i)}};return depth,parent,pred,nil}

func sha(path string)(string,int64,error){f,err:=os.Open(path);if err!=nil{return "",0,err};defer f.Close();h:=sha256.New();n,err:=io.Copy(h,f);return hex.EncodeToString(h.Sum(nil)),n,err}
func phaseGenerators(phase int)[]string{faceCount:=map[int]int{3:7,4:6,5:5,6:4}[phase];result:=make([]string,0,faceCount*4);for face:=0;face<faceCount;face++{for power:=1;power<=4;power++{result=append(result,fmt.Sprintf("%s%d",faces[face],power))}};return result}
func publish(phase int,out string,depth,parent []byte,pred []uint32) error {if err:=os.MkdirAll(out,0755);err!=nil{return err};if err:=atomicWrite(filepath.Join(out,"depths.bin"),depth);err!=nil{return err};if err:=atomicWrite(filepath.Join(out,"predecessors.bin"),parent);err!=nil{return err};hist:=map[int]int{};diameter:=-1;reachable:=0;for _,d:=range depth{if d!=unseen{hist[int(d)]++;reachable++;if int(d)>diameter{diameter=int(d)}}};anti:=make([]uint32,0,hist[diameter]);for i,d:=range depth{if int(d)==diameter{anti=append(anti,uint32(i))}};if err:=atomicWrite(filepath.Join(out,"antipodes.bin"),encodeFrontier(anti));err!=nil{return err};var lines strings.Builder;lines.WriteString("depth,count\n");for d:=0;d<=diameter;d++{fmt.Fprintf(&lines,"%d,%d\n",d,hist[d])};if err:=atomicWrite(filepath.Join(out,"histogram.csv"),[]byte(lines.String()));err!=nil{return err}
	files:=[]string{"depths.bin","predecessors.bin","antipodes.bin","histogram.csv"};if pred!=nil{if err:=atomicWrite(filepath.Join(out,"predecessor_indices.bin"),encodeFrontier(pred));err!=nil{return err};files=append(files,"predecessor_indices.bin")};payloads:=map[string]any{};for _,name:=range files{sum,n,err:=sha(filepath.Join(out,name));if err!=nil{return err};payloads[name]=map[string]any{"sha256":sum,"bytes":n}}
	transition:=map[int]string{3:"G5->G6",4:"G6->G7",5:"G7->G8",6:"G8->G9"}[phase];meta:=map[string]any{"schema_version":1,"phase":phase,"transition":transition,"repository_commit":upstreamCommit,"upstream_commit":upstreamCommit,"metric":"FTM","generators":phaseGenerators(phase),"state_count":reachable,"storage_slots":len(depth),"diameter":diameter,"antipode_count":len(anti),"index_encoding":map[bool]string{true:"upstream-hash6-sparse",false:"dense-component-v1"}[phase==4],"predecessor_encoding":"one deterministic lowering move code; 255 only for solved/unreachable","payloads":payloads,"complete":true};if err:=atomicJSON(filepath.Join(out,"metadata.json"),meta);err!=nil{return err};fmt.Printf("phase %d complete: states=%d diameter=%d antipodes=%d\n",phase,reachable,diameter,len(anti));return nil}

func estimate(){for _,p:=range []int{3,4,5,6}{slots:=map[int]int{3:208099584,4:103224,5:64157184,6:25945920}[p];persistent:=2*slots+4*map[int]int{3:212,4:2531,5:3484,6:117}[p];fmt.Printf("phase %d: %.3f GiB persistent\n",p,float64(persistent)/(1<<30))};fmt.Println("all phases: approximately 0.556 GiB plus metadata; active checkpoint temporarily duplicates one phase")}

func encodeFullState(s totalState) []byte {b:=make([]byte,108);copy(b,[]byte("MDRFSV1\x00"));for i:=0;i<30;i++{b[8+i]=s.edgePositions[i].piece;if s.edgePositions[i].orientation{b[38+i]=1}};for i:=0;i<20;i++{b[68+i]=s.cornerPositions[i].piece;b[88+i]=s.cornerPositions[i].orientation};return b}
func readU32(path string)([]uint32,error){b,err:=os.ReadFile(path);if err!=nil{return nil,err};return decodeFrontier(b)}

type extractor struct {phase,faceCount int;depth,parent []byte;pred []uint32;l *layout;root uint32;repCache map[uint32]totalState}
func (x *extractor) next(index uint32,code int) uint32 {if x.l!=nil{return x.l.next(index,code)};s:=x.representative(index);return hash6Gen(applyFull(s,code))}
func (x *extractor) loweringWord(index uint32)([]byte,error){word:=make([]byte,0,16);for x.depth[index]>0{code:=x.parent[index];if code==unseen||int(code)>=x.faceCount*4{return nil,fmt.Errorf("invalid predecessor move at %d",index)};word=append(word,code);if x.l!=nil{index=x.l.next(index,int(code))}else{if x.pred[index]==^uint32(0){return nil,fmt.Errorf("missing predecessor index")};index=x.pred[index]}};if index!=x.root{return nil,fmt.Errorf("predecessor path misses root")};return word,nil}
func (x *extractor) representative(index uint32) totalState {if s,ok:=x.repCache[index];ok{return s};word,err:=x.loweringWord(index);if err!=nil{panic(err)};s:=defaultState;for i:=len(word)-1;i>=0;i--{s=applyFull(s,int(inverseCode(int(word[i]))))};if x.repCache!=nil{x.repCache[index]=s};return s}
func (x *extractor) loweringMask(index uint32)uint32{d:=x.depth[index];var mask uint32;for code:=0;code<x.faceCount*4;code++{n:=x.next(index,code);if x.depth[n]!=unseen&&x.depth[n]+1==d{mask|=uint32(1)<<code}};return mask}

func extractHard(phase int,table,out string) error {depth,err:=os.ReadFile(filepath.Join(table,"depths.bin"));if err!=nil{return err};parent,err:=os.ReadFile(filepath.Join(table,"predecessors.bin"));if err!=nil{return err};if len(depth)!=len(parent){return errors.New("table payload length mismatch")};x:=extractor{phase:phase,depth:depth,parent:parent,repCache:map[uint32]totalState{}}
	if phase==4{x.faceCount=6;x.root=hash6Gen(defaultState);x.pred,err=readU32(filepath.Join(table,"predecessor_indices.bin"));if err!=nil{return err};if len(x.pred)!=len(depth){return errors.New("predecessor index length mismatch")}}else{x.l,err=phaseLayout(phase);if err!=nil{return err};x.faceCount=x.l.faceCount;x.root=x.l.denseFromState(defaultState)}
	diameter:=byte(0);for _,d:=range depth{if d!=unseen&&d>diameter{diameter=d}};indices:=make([]uint32,0);for i,d:=range depth{if d==diameter{indices=append(indices,uint32(i))}}
	lastMemo:=make([]uint32,len(depth));var lastMask func(uint32)uint32;lastMask=func(index uint32)uint32{if lastMemo[index]!=0{return lastMemo[index]};d:=depth[index];first:=x.loweringMask(index);var result uint32;for code:=0;code<x.faceCount*4;code++{if first&(uint32(1)<<code)==0{continue};if d==1{result|=uint32(1)<<code}else{result|=lastMask(x.next(index,code))}};lastMemo[index]=result;return result}
	var payload bytes.Buffer;payload.Write([]byte("MDRHSV1\x00"));payload.WriteByte(byte(phase));payload.WriteByte(diameter);binary.Write(&payload,binary.LittleEndian,uint16(172));binary.Write(&payload,binary.LittleEndian,uint32(len(indices)))
	for _,index:=range indices{word,err:=x.loweringWord(index);if err!=nil{return err};s:=x.representative(index);observed:=upstreamHash(phase,s);if x.l!=nil{if x.l.upstreamFromDense(index)!=observed{return fmt.Errorf("representative coordinate mismatch %d",index)}}else if observed!=index{return fmt.Errorf("representative coordinate mismatch %d",index)};solved:=s;for _,code:=range word{solved=applyFull(solved,int(code))};if encode:=encodeFullState(solved);!bytes.Equal(encode,encodeFullState(defaultState)){return fmt.Errorf("solution replay failed %d",index)}
		record:=make([]byte,140);binary.LittleEndian.PutUint32(record[0:4],index);record[4]=diameter;record[5]=byte(len(word));copy(record[8:116],encodeFullState(s));for i:=116;i<132;i++{record[i]=unseen};copy(record[116:132],word);binary.LittleEndian.PutUint32(record[132:136],x.loweringMask(index));binary.LittleEndian.PutUint32(record[136:140],lastMask(index));digest:=sha256.Sum256(record);payload.Write(record);payload.Write(digest[:])}
	if err:=atomicWrite(out,payload.Bytes());err!=nil{return err};sum,n,err:=sha(out);if err!=nil{return err};manifest:=map[string]any{"schema_version":1,"phase":phase,"depth":diameter,"count":len(indices),"record_size":172,"format":"MDRHSV1","table_metadata":filepath.Join(table,"metadata.json"),"payload":map[string]any{"file":filepath.Base(out),"bytes":n,"sha256":sum},"complete":true};if err:=atomicJSON(out+".metadata.json",manifest);err!=nil{return err};fmt.Printf("phase %d hard states: %d depth %d, all coordinates and solutions replayed\n",phase,len(indices),diameter);return nil}

type hardRecord struct {index uint32;depth byte;solution []byte;state [108]byte}
func readHardGo(path string)(byte,[]hardRecord,error){data,err:=os.ReadFile(path);if err!=nil{return 0,nil,err};if len(data)<16||string(data[:8])!="MDRHSV1\x00"{return 0,nil,errors.New("bad hard-state header")};phase:=data[8];recordSize:=int(binary.LittleEndian.Uint16(data[10:12]));count:=int(binary.LittleEndian.Uint32(data[12:16]));if recordSize!=172||len(data)!=16+count*recordSize{return 0,nil,errors.New("bad hard-state length")};result:=make([]hardRecord,count);for i:=0;i<count;i++{off:=16+i*recordSize;body:=data[off:off+140];sum:=sha256.Sum256(body);if !bytes.Equal(sum[:],data[off+140:off+172]){return 0,nil,fmt.Errorf("hard-state record checksum %d",i)};length:=int(body[5]);if length>16{return 0,nil,errors.New("hard-state solution length")};result[i].index=binary.LittleEndian.Uint32(body[:4]);result[i].depth=body[4];result[i].solution=append([]byte(nil),body[116:116+length]...);copy(result[i].state[:],body[8:116])};return phase,result,nil}
func representativeFromSolution(solution []byte) totalState{s:=defaultState;for i:=len(solution)-1;i>=0;i--{s=applyFull(s,int(inverseCode(int(solution[i]))))};return s}

func composePair(name,hardA,hardB,out string) error {phaseA,a,err:=readHardGo(hardA);if err!=nil{return err};phaseB,b,err:=readHardGo(hardB);if err!=nil{return err};expected:=map[string][3]int{"pair34":{3,4,536572},"pair56":{5,6,407628}};control,ok:=expected[name];if !ok||int(phaseA)!=control[0]||int(phaseB)!=control[1]{return errors.New("pair and hard-state phases disagree")};raw:=len(a)*len(b);if raw!=control[2]{return fmt.Errorf("raw pair count %d differs from control %d",raw,control[2])};la,_:=phaseLayout(int(phaseA));var lb *layout;if phaseB!=4{lb,_=phaseLayout(int(phaseB))}
	repsA:=make([]totalState,len(a));for i,r:=range a{repsA[i]=representativeFromSolution(r.solution);if !bytes.Equal(encodeFullState(repsA[i]),r.state[:]){return fmt.Errorf("phase A representative mismatch %d",i)}};repsB:=make([]totalState,len(b));for i,r:=range b{repsB[i]=representativeFromSolution(r.solution);if !bytes.Equal(encodeFullState(repsB[i]),r.state[:]){return fmt.Errorf("phase B representative mismatch %d",i)}}
	rawStates:=make([]byte,0,raw*108);pairIndices:=make([]byte,0,raw*8);rawToUnique:=make([]uint32,0,raw);unique:=make([]byte,0,raw*108);ids:=make(map[[108]byte]uint32,raw);multiplicity:=make([]uint32,0,raw);valid:=0
	for ai,ar:=range a{for bi,br:=range b{s:=repsB[bi];forwardA:=ar.solution;for i:=len(forwardA)-1;i>=0;i--{s=applyFull(s,int(inverseCode(int(forwardA[i]))))};if la.denseFromState(s)!=ar.index{return fmt.Errorf("first coordinate mismatch pair %d,%d",ai,bi)};after:=s;for _,code:=range ar.solution{after=applyFull(after,int(code))};if !bytes.Equal(encodeFullState(after),br.state[:]){return fmt.Errorf("boundary state mismatch pair %d,%d",ai,bi)};var observedB uint32;if phaseB==4{observedB=hash6Gen(after)}else{observedB=lb.denseFromState(after)};if observedB!=br.index{return fmt.Errorf("second coordinate mismatch pair %d,%d",ai,bi)};encoded:=encodeFullState(s);rawStates=append(rawStates,encoded...);var pair [8]byte;binary.LittleEndian.PutUint32(pair[:4],ar.index);binary.LittleEndian.PutUint32(pair[4:],br.index);pairIndices=append(pairIndices,pair[:]...);var key [108]byte;copy(key[:],encoded);id,exists:=ids[key];if !exists{id=uint32(len(ids));ids[key]=id;unique=append(unique,encoded...);multiplicity=append(multiplicity,0)};multiplicity[id]++;rawToUnique=append(rawToUnique,id);valid++}}
	maxMultiplicity:=uint32(0);for _,m:=range multiplicity{if m>maxMultiplicity{maxMultiplicity=m}};if err:=os.MkdirAll(out,0755);err!=nil{return err};payloadData:=map[string][]byte{"states.bin":rawStates,"pair_indices.bin":pairIndices,"raw_to_unique.bin":encodeFrontier(rawToUnique),"unique_states.bin":unique};payloads:=map[string]any{};for name,data:=range payloadData{path:=filepath.Join(out,name);if err:=atomicWrite(path,data);err!=nil{return err};sum,n,err:=sha(path);if err!=nil{return err};payloads[name]=map[string]any{"sha256":sum,"bytes":n}}
	stats:=map[string]any{"pair":name,"raw_pairs":raw,"valid_pairs":valid,"unique_full_states":len(ids),"duplicate_pairs":raw-len(ids),"maximum_multiplicity":maxMultiplicity};if err:=atomicJSON(filepath.Join(out,"statistics.json"),stats);err!=nil{return err};sum,n,err:=sha(filepath.Join(out,"statistics.json"));if err!=nil{return err};payloads["statistics.json"]=map[string]any{"sha256":sum,"bytes":n};manifest:=map[string]any{"schema_version":1,"pair":name,"phase_order":[]int{int(phaseA),int(phaseB)},"composition_order":"representative(second) then representative(first), required by left cosets Hs","raw_pairs":raw,"valid_pairs":valid,"unique_full_states":len(ids),"payloads":payloads,"complete":true};if err:=atomicJSON(filepath.Join(out,"metadata.json"),manifest);err!=nil{return err};fmt.Printf("%s complete: raw=%d valid=%d unique=%d duplicates=%d max_multiplicity=%d\n",name,raw,valid,len(ids),raw-len(ids),maxMultiplicity);return nil}

func facesCommute(a,b int)bool{if a==b{return false};for _,ca:=range specs[a].c{for _,cb:=range specs[b].c{if ca==cb{return false}}};for _,ea:=range specs[a].e{for _,eb:=range specs[b].e{if ea==eb{return false}}};return true}
func normalizeWord(input []byte)[]byte{word:=append([]byte(nil),input...);for{changed:=false;for i:=0;i+1<len(word);i++{fa,fb:=int(word[i])/4,int(word[i+1])/4;if fa==fb{power:=(int(word[i])%4+1+int(word[i+1])%4+1)%5;if power==0{word=append(word[:i],word[i+2:]...)}else{word[i]=byte(fa*4+power-1);word=append(word[:i+1],word[i+2:]...)};changed=true;break};if facesCommute(fa,fb)&&fa>fb{word[i],word[i+1]=word[i+1],word[i];changed=true;break}};if !changed{return word}}}
type shortNode struct{state totalState;word []byte}
func composeTotal(left,right totalState)totalState{result:=totalState{};for d:=0;d<30;d++{source:=int(right.edgePositions[d].piece);piece:=left.edgePositions[source].piece;orientation:=left.edgePositions[source].orientation!=right.edgePositions[d].orientation;result.edgePositions[d]=edgePosition{piece,orientation};result.edgePieces[piece]=uint8(d)};for d:=0;d<20;d++{source:=int(right.cornerPositions[d].piece);piece:=left.cornerPositions[source].piece;orientation:=(left.cornerPositions[source].orientation+right.cornerPositions[d].orientation)%3;result.cornerPositions[d]=cornerPosition{piece,orientation};result.cornerPieces[piece]=uint8(d)};return result}
func wordState(word []byte)totalState{s:=defaultState;for _,code:=range word{s=applyFull(s,int(code))};return s}
func inverseWordState(word []byte)totalState{s:=defaultState;for i:=len(word)-1;i>=0;i--{s=applyFull(s,int(inverseCode(int(word[i]))))};return s}
func shortNodes(faceCount,maxDepth int)[]shortNode{all:=[]shortNode{{defaultState,nil}};frontier:=all;seen:=map[[108]byte]bool{};var root [108]byte;copy(root[:],encodeFullState(defaultState));seen[root]=true;for depth:=1;depth<=maxDepth;depth++{next:=make([]shortNode,0,len(frontier)*faceCount*3);for _,node:=range frontier{lastFace:=-1;if len(node.word)>0{lastFace=int(node.word[len(node.word)-1])/4};for code:=0;code<faceCount*4;code++{if code/4==lastFace{continue};s:=applyFull(node.state,code);var key [108]byte;copy(key[:],encodeFullState(s));if seen[key]{continue};seen[key]=true;word:=append(append([]byte(nil),node.word...),byte(code));next=append(next,shortNode{s,word})}};all=append(all,next...);frontier=next};return all}
func shortDictionary(faceCount,maxDepth int)map[[108]byte][]byte{rootKey:=[108]byte{};copy(rootKey[:],encodeFullState(defaultState));result:=map[[108]byte][]byte{rootKey:{}};frontier:=[]shortNode{{defaultState,nil}};for depth:=1;depth<=maxDepth;depth++{next:=make([]shortNode,0,len(frontier)*faceCount*3);for _,node:=range frontier{lastFace:=-1;if len(node.word)>0{lastFace=int(node.word[len(node.word)-1])/4};for code:=0;code<faceCount*4;code++{if code/4==lastFace{continue};s:=applyFull(node.state,code);var key [108]byte;copy(key[:],encodeFullState(s));if _,seen:=result[key];seen{continue};word:=append(append([]byte(nil),node.word...),byte(code));result[key]=word;next=append(next,shortNode{s,word})}};frontier=next};return result}
func boundaryRewrite(a,b []byte)([]byte,bool){if len(a)==0||len(b)==0{return nil,false};left,right:=a[len(a)-1],b[0];if left/4!=right/4{return nil,false};power:=(int(left)%4+1+int(right)%4+1)%5;word:=append([]byte(nil),a[:len(a)-1]...);if power!=0{word=append(word,byte(int(left/4)*4+power-1))};word=append(word,b[1:]...);return word,true}
func k3Rewrites(a,b []hardRecord,faceCount int)map[string][]byte{suffixes:=map[string][]byte{};prefixes:=map[string][]byte{};for _,r:=range a{w:=append([]byte(nil),r.solution[len(r.solution)-3:]...);suffixes[string(w)]=w};for _,r:=range b{w:=append([]byte(nil),r.solution[:3]...);prefixes[string(w)]=w};xs:=shortNodes(faceCount,2);ys:=shortNodes(faceCount,3);type candidate struct{key [108]byte;word []byte};left:=map[string][]candidate{};for name,aw:=range suffixes{invA:=inverseWordState(aw);rows:=make([]candidate,0,len(xs));for _,x:=range xs{s:=composeTotal(invA,x.state);var key [108]byte;copy(key[:],encodeFullState(s));rows=append(rows,candidate{key,x.word})};left[name]=rows};invY:=make([]totalState,len(ys));for i,y:=range ys{invY[i]=inverseWordState(y.word)};result:=map[string][]byte{};for bname,bw:=range prefixes{bstate:=wordState(bw);right:=make(map[[108]byte][]byte,len(ys));for i,y:=range ys{s:=composeTotal(bstate,invY[i]);var key [108]byte;copy(key[:],encodeFullState(s));if old,ok:=right[key];!ok||len(y.word)<len(old){right[key]=y.word}};for aname,rows:=range left{var best []byte;for _,row:=range rows{if y,ok:=right[row.key];ok&&len(row.word)+len(y)<6&&(best==nil||len(row.word)+len(y)<len(best)){best=append(append([]byte(nil),row.word...),y...)}};if best!=nil{result[aname+bname]=best}}};fmt.Printf("local k=3 MITM faces=%d: suffixes=%d prefixes=%d reducible_windows=%d\n",faceCount,len(suffixes),len(prefixes),len(result));return result}
func k4Rewrites(a,b []hardRecord,faceCount int)map[string][]byte{suffixes:=map[string][]byte{};prefixes:=map[string][]byte{};for _,r:=range a{w:=append([]byte(nil),r.solution[len(r.solution)-4:]...);suffixes[string(w)]=w};for _,r:=range b{w:=append([]byte(nil),r.solution[:4]...);prefixes[string(w)]=w};xs:=shortNodes(faceCount,3);ys:=shortNodes(faceCount,4);type candidate struct{key [108]byte;word []byte};left:=map[string][]candidate{};for name,aw:=range suffixes{invA:=inverseWordState(aw);rows:=make([]candidate,0,len(xs));for _,x:=range xs{s:=composeTotal(invA,x.state);var key [108]byte;copy(key[:],encodeFullState(s));rows=append(rows,candidate{key,x.word})};left[name]=rows};invY:=make([]totalState,len(ys));for i,y:=range ys{invY[i]=inverseWordState(y.word)};result:=map[string][]byte{};for bname,bw:=range prefixes{bstate:=wordState(bw);right:=make(map[[108]byte][]byte,len(ys));for i,y:=range ys{s:=composeTotal(bstate,invY[i]);var key [108]byte;copy(key[:],encodeFullState(s));if old,ok:=right[key];!ok||len(y.word)<len(old){right[key]=y.word}};for aname,rows:=range left{var best []byte;for _,row:=range rows{if y,ok:=right[row.key];ok&&len(row.word)+len(y)<8&&(best==nil||len(row.word)+len(y)<len(best)){best=append(append([]byte(nil),row.word...),y...)}};if best!=nil{result[aname+bname]=best}}};fmt.Printf("local k=4 MITM faces=%d: suffixes=%d prefixes=%d reducible_windows=%d\n",faceCount,len(suffixes),len(prefixes),len(result));return result}
type joinRecord struct{hash uint64;group uint16;node uint32}
func stateHash64(s totalState)uint64{h:=uint64(1469598103934665603);mix:=func(v byte){h^=uint64(v);h*=1099511628211};for _,p:=range s.edgePositions{mix(p.piece);if p.orientation{mix(1)}else{mix(0)}};for _,p:=range s.cornerPositions{mix(p.piece);mix(p.orientation)};return h}
func k4JoinRewrites(a,b []hardRecord,faceCount int)map[string][]byte{suffixSet:=map[string]bool{};prefixSet:=map[string]bool{};for _,r:=range a{suffixSet[string(r.solution[len(r.solution)-4:])]=true};for _,r:=range b{prefixSet[string(r.solution[:4])]=true};suffixes:=make([]string,0,len(suffixSet));prefixes:=make([]string,0,len(prefixSet));for w:=range suffixSet{suffixes=append(suffixes,w)};for w:=range prefixSet{prefixes=append(prefixes,w)};sort.Strings(suffixes);sort.Strings(prefixes);xs:=shortNodes(faceCount,4);ys:=shortNodes(faceCount,3);invA:=make([]totalState,len(suffixes));for i,w:=range suffixes{invA[i]=inverseWordState([]byte(w))};invY:=make([]totalState,len(ys));for i,y:=range ys{invY[i]=inverseWordState(y.word)};const bucketBits=12;var buckets [1<<bucketBits][]joinRecord;for group:=range suffixes{for node,x:=range xs{s:=composeTotal(invA[group],x.state);h:=stateHash64(s);bucket:=h>>(64-bucketBits);buckets[bucket]=append(buckets[bucket],joinRecord{h,uint16(group),uint32(node)})}};for i:=range buckets{sort.Slice(buckets[i],func(a,b int)bool{return buckets[i][a].hash<buckets[i][b].hash})};result:=map[string][]byte{};for bgroup,bw:=range prefixes{bstate:=wordState([]byte(bw));for ynode,y:=range ys{right:=composeTotal(bstate,invY[ynode]);h:=stateHash64(right);bucket:=buckets[h>>(64-bucketBits)];start:=sort.Search(len(bucket),func(i int)bool{return bucket[i].hash>=h});for i:=start;i<len(bucket)&&bucket[i].hash==h;i++{row:=bucket[i];left:=composeTotal(invA[row.group],xs[row.node].state);if left!=right{continue};replacement:=append(append([]byte(nil),xs[row.node].word...),y.word...);if len(replacement)>=8{continue};key:=suffixes[row.group]+prefixes[bgroup];if old,ok:=result[key];!ok||len(replacement)<len(old){result[key]=replacement}}}};fmt.Printf("local k=4 balanced MITM faces=%d: suffixes=%d prefixes=%d radius4=%d radius3=%d reducible_windows=%d\n",faceCount,len(suffixes),len(prefixes),len(xs),len(ys),len(result));return result}
var activeK3,activeK4 map[string][]byte
func localRewrite(a,b []byte,short map[[108]byte][]byte,bound int)([]byte,bool){base:=append(append([]byte(nil),a...),b...);normal:=normalizeWord(base);if len(normal)<=bound{return normal,true};if len(a)>=2&&len(b)>=2{window:=append(append([]byte(nil),a[len(a)-2:]...),b[:2]...);s:=wordState(window);var key [108]byte;copy(key[:],encodeFullState(s));if replacement,ok:=short[key];ok&&len(replacement)<4{word:=append([]byte(nil),a[:len(a)-2]...);word=append(word,replacement...);word=append(word,b[2:]...);if len(word)<=bound{return word,true}}};if len(a)>=3&&len(b)>=3{window:=string(append(append([]byte(nil),a[len(a)-3:]...),b[:3]...));if replacement,ok:=activeK3[window];ok{word:=append([]byte(nil),a[:len(a)-3]...);word=append(word,replacement...);word=append(word,b[3:]...);if len(word)<=bound{return word,true}}};if len(a)>=4&&len(b)>=4{window:=string(append(append([]byte(nil),a[len(a)-4:]...),b[:4]...));if replacement,ok:=activeK4[window];ok{word:=append([]byte(nil),a[:len(a)-4]...);word=append(word,replacement...);word=append(word,b[4:]...);if len(word)<=bound{return word,true}}};return nil,false}

func reducePair(name,hardA,hardB,out string)error{phaseA,a,err:=readHardGo(hardA);if err!=nil{return err};phaseB,b,err:=readHardGo(hardB);if err!=nil{return err};controls:=map[string][4]int{"pair34":{3,4,536572,21},"pair56":{5,6,407628,25}};control,ok:=controls[name];if !ok||int(phaseA)!=control[0]||int(phaseB)!=control[1]||len(a)*len(b)!=control[2]{return errors.New("reduction input controls mismatch")};bound:=control[3];repsA:=make([]totalState,len(a));for i,r:=range a{repsA[i]=representativeFromSolution(r.solution)};repsB:=make([]totalState,len(b));for i,r:=range b{repsB[i]=representativeFromSolution(r.solution)}
	// The source generator count is 7 for phase 3 and 5 for phase 5.
	faceCount:=map[int]int{3:7,5:5}[control[0]];if configured:=os.Getenv("MDR_LOCAL_FACE_COUNT");configured!=""{parsed,parseErr:=strconv.Atoi(configured);if parseErr!=nil||parsed<faceCount||parsed>12{return errors.New("MDR_LOCAL_FACE_COUNT must be between the phase source count and 12")};faceCount=parsed};short:=shortDictionary(faceCount,3);activeK3=k3Rewrites(a,b,faceCount);activeK4=nil;if control[0]==5{activeK4=k4Rewrites(a,b,faceCount)}else if control[0]==3&&faceCount==7{activeK4=k4JoinRewrites(a,b,faceCount)};raw:=len(a)*len(b);mapping:=make([]byte,raw*16);witnesses:=make([]byte,0);remaining:=make([]uint32,0,raw);boundaryCount,localCount:=0,0;rawID:=0;maxVerified:=0
	for _,ar:=range a{for bi,br:=range b{var witness []byte;status:=byte(0);if w,yes:=boundaryRewrite(ar.solution,br.solution);yes&&len(w)<=bound{witness,status=w,1}else if w,yes:=localRewrite(ar.solution,br.solution,short,bound);yes{witness,status=w,2};record:=mapping[rawID*16:(rawID+1)*16];binary.LittleEndian.PutUint32(record[0:4],uint32(rawID));if status==0{record[4]=0;record[5]=0;binary.LittleEndian.PutUint32(record[8:12],uint32(rawID));remaining=append(remaining,uint32(rawID))}else{s:=repsB[bi];for i:=len(ar.solution)-1;i>=0;i--{s=applyFull(s,int(inverseCode(int(ar.solution[i]))))};for _,code:=range witness{s=applyFull(s,int(code))};if !bytes.Equal(encodeFullState(s),encodeFullState(defaultState)){return fmt.Errorf("reduction witness failed raw %d",rawID)};record[4]=status;record[5]=byte(len(witness));binary.LittleEndian.PutUint32(record[8:12],^uint32(0));binary.LittleEndian.PutUint32(record[12:16],uint32(len(witnesses)));witnesses=append(witnesses,witness...);if len(witness)>maxVerified{maxVerified=len(witness)};if status==1{boundaryCount++}else{localCount++}};rawID++}}
	if err:=os.MkdirAll(out,0755);err!=nil{return err};payloadData:=map[string][]byte{"mapping.bin":mapping,"witnesses.bin":witnesses,"remaining_ids.bin":encodeFrontier(remaining)};payloads:=map[string]any{};for file,data:=range payloadData{path:=filepath.Join(out,file);if err:=atomicWrite(path,data);err!=nil{return err};sum,n,err:=sha(path);if err!=nil{return err};payloads[file]=map[string]any{"sha256":sum,"bytes":n}};maxK:=3;if activeK4!=nil{maxK=4};stats:=map[string]any{"pair":name,"raw_pairs":raw,"unique_full_states":raw,"symmetry_orbits":raw,"inversion_orbits":nil,"inversion_applied":false,"closed_by_boundary_merge":boundaryCount,"closed_by_local_rewrite":localCount,"remaining_canonical_representatives":len(remaining),"largest_orbit":1,"median_orbit":1,"maximum_verified_witness_length":maxVerified,"local_search_window_k":maxK,"local_search_face_count":faceCount};if err:=atomicJSON(filepath.Join(out,"statistics.json"),stats);err!=nil{return err};sum,n,err:=sha(filepath.Join(out,"statistics.json"));if err!=nil{return err};payloads["statistics.json"]=map[string]any{"sha256":sum,"bytes":n};meta:=map[string]any{"schema_version":1,"pair":name,"raw_pairs":raw,"max_length":bound,"mapping_record_bytes":16,"status_codes":map[string]int{"remaining":0,"boundary":1,"local":2},"symmetry_policy":"identity pending exhaustive rotation stabilizer report","inversion_policy":"not applied pending exact analysis","local_rewrite_policy":fmt.Sprintf("commuting normalization and exact MITM boundary windows through k=%d over %d faces",maxK,faceCount),"payloads":payloads,"complete":true};if err:=atomicJSON(filepath.Join(out,"metadata.json"),meta);err!=nil{return err};fmt.Printf("%s reductions: boundary=%d local=%d remaining=%d\n",name,boundaryCount,localCount,len(remaining));return nil}

func main(){if len(os.Args)<2{fmt.Fprintln(os.Stderr,"usage: mdr-table <self-test|estimate|build|extract|compose|reduce>");os.Exit(2)};var err error;switch os.Args[1]{case "self-test":err=selfTest();case "estimate":estimate();case "reduce":f:=flag.NewFlagSet("reduce",flag.ContinueOnError);pair:=f.String("pair","","pair34 or pair56");a:=f.String("hard-a","","first hard-state file");b:=f.String("hard-b","","second hard-state file");out:=f.String("out","","reduction directory");if e:=f.Parse(os.Args[2:]);e!=nil{err=e;break};if *pair==""||*a==""||*b==""||*out==""||f.NArg()!=0{err=errors.New("reduce requires --pair --hard-a --hard-b --out")}else{err=reducePair(*pair,*a,*b,*out)};case "compose":f:=flag.NewFlagSet("compose",flag.ContinueOnError);pair:=f.String("pair","","pair34 or pair56");a:=f.String("hard-a","","first hard-state file");b:=f.String("hard-b","","second hard-state file");out:=f.String("out","","composition directory");if e:=f.Parse(os.Args[2:]);e!=nil{err=e;break};if *pair==""||*a==""||*b==""||*out==""||f.NArg()!=0{err=errors.New("compose requires --pair --hard-a --hard-b --out")}else{err=composePair(*pair,*a,*b,*out)};case "extract":f:=flag.NewFlagSet("extract",flag.ContinueOnError);phase:=f.Int("phase",0,"3..6");table:=f.String("table","","table directory");out:=f.String("out","","hard-state file");if e:=f.Parse(os.Args[2:]);e!=nil{err=e;break};if *phase<3||*phase>6||*table==""||*out==""||f.NArg()!=0{err=errors.New("extract requires --phase 3..6 --table DIR --out FILE")}else{err=extractHard(*phase,*table,*out)};case "build":f:=flag.NewFlagSet("build",flag.ContinueOnError);phase:=f.Int("phase",0,"3..6");out:=f.String("out","","output directory");workers:=f.Int("workers",10,"1..10");resume:=f.Bool("resume",false,"resume checkpoint");if e:=f.Parse(os.Args[2:]);e!=nil{err=e;break};if *out==""||*phase<3||*phase>6||*workers<1||*workers>10||f.NArg()!=0{err=errors.New("build requires --phase 3..6 --out DIR and workers 1..10");break};runtime.GOMAXPROCS(*workers);var d,p []byte;var pred []uint32;if *phase==4{d,p,pred,err=buildSparse4(*out)}else{var l *layout;l,err=phaseLayout(*phase);if err==nil{d,p,err=buildDense(l,*out,*workers,*resume)}};if err==nil{err=publish(*phase,*out,d,p,pred)};if err==nil{err=os.RemoveAll(filepath.Join(*out,".work"))};default:err=fmt.Errorf("unknown command %q",os.Args[1])};if err!=nil{fmt.Fprintln(os.Stderr,"error:",err);os.Exit(2)}}
