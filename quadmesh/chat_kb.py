"""Verified engineering / printing / Quadmesh facts. Used three ways: (1) answers when no chat model is trained yet (keyword
retrieval), (2) training data for the chat model, (3) a source of truth the model's answers can be checked against."""
from __future__ import annotations
import re
from typing import List, Optional, Tuple

# (keywords, [questions], answer)
FACTS: List[Tuple[str, List[str], str]] = [
 ("pla material", ["what is PLA", "is PLA good for strong parts", "should I use PLA"],
  "PLA is easy to print and stiff, but brittle, it creeps under long loads and softens around 55 C. Good for models and light parts, not for hot or load-bearing ones."),
 ("petg material", ["what is PETG", "why use PETG", "is PETG strong"],
  "PETG is tougher than PLA, handles more heat and is a good general-purpose choice for functional parts. It strings a bit and sticks hard to smooth beds."),
 ("abs asa material", ["what is ABS", "what is ASA", "when should I use ABS"],
  "ABS and ASA take more heat and impact than PLA but warp and need an enclosed printer and ventilation. ASA also resists sunlight, so it suits outdoor parts."),
 ("nylon pa12 carbon material", ["what is nylon", "what is carbon fibre nylon", "when to use PA12-CF"],
  "Nylon is tough and wear resistant but absorbs moisture, so dry it first. Carbon-fibre nylon is stiffer and stronger and needs a hardened nozzle."),
 ("wall thickness minimum", ["how thin can a wall be", "what is the minimum wall thickness", "what wall thickness should I use"],
  "With a 0.4 mm nozzle the thinnest reliable wall is about 0.8 mm (two perimeters). For parts that carry load use 2 mm or more."),
 ("overhang support angle", ["what is an overhang", "when do I need supports", "how steep can an overhang be"],
  "Surfaces steeper than about 45 degrees from vertical droop without support. Quadmesh reports the percentage of overhanging surface; below your limit it prints support-free."),
 ("hole tolerance clearance fit", ["why are my printed holes too small", "how much clearance for a pin", "what clearance for moving parts"],
  "Printed holes come out slightly small. Add about 0.2 to 0.4 mm to a hole diameter for a sliding fit; Quadmesh uses 0.3 mm on link pin holes and nut threads."),
 ("layer strength orientation", ["which direction is a 3D print strongest", "why did my print snap between layers", "print orientation strength"],
  "A print is much weaker across layers than along them, often by half. Orient the part so loads run along the layers, not pulling them apart."),
 ("fillet stress corner", ["why add fillets", "what does a fillet do", "should I round corners"],
  "Sharp inside corners concentrate stress and crack first. A fillet spreads the stress, so rounded corners make stronger printed parts."),
 ("gear module teeth", ["what is gear module", "how do I pick a gear module", "what is the pitch diameter of a gear"],
  "Module is the tooth size in mm: pitch diameter = module x teeth. Two gears mesh only if they have the same module and a centre distance of module x (z1+z2) / 2."),
 ("backlash gear", ["what is backlash", "why do gears need backlash", "how much backlash for printed gears"],
  "Backlash is the small gap between meshing teeth. Printed gears need about 0.15 to 0.3 mm so they do not bind; Quadmesh thins each tooth by that amount."),
 ("thread pitch metric", ["what is thread pitch", "what is M8 pitch", "how do metric threads work"],
  "Pitch is the distance between thread crests. Coarse metric pitches: M3 0.5, M4 0.7, M5 0.8, M6 1.0, M8 1.25, M10 1.5. Printed threads work best at M6 and larger."),
 ("thread printing vertical", ["how do I print threads", "can I 3D print a bolt", "how to print a nut"],
  "Print bolts standing upright so the thread is round, and leave 0.3 mm clearance in printed nuts. Threads under M6 are fragile; consider metal hardware for strength."),
 ("step file brep", ["what is a STEP file", "STEP versus STL", "why use STEP"],
  "STEP stores exact CAD geometry (B-Rep) that CAD programs can edit; STL is only a mesh of triangles for printing. Quadmesh reads and writes STEP through the CadQuery kernel."),
 ("stl format", ["what is an STL file", "what does watertight mean", "why must a mesh be watertight"],
  "STL is a triangle mesh. Watertight means every edge is shared by exactly two triangles, with no holes, so a slicer knows what is inside the part."),
 ("photo to 3d limits", ["can you make a 3D model from a photo", "how does photo to 3D work", "how accurate is image to 3D"],
  "From one photo Quadmesh guesses: symmetric upright objects are spun into a solid of revolution, other shapes are inflated from their outline. The back and depth are invented, so check before printing."),
 ("what can quadmesh build", ["what can you make", "what can Quadmesh do", "what can you build"],
  "I make parts from dimensions (plates, brackets, flanges, enclosures), gears, threads, springs, pipes, vases, 3D text, four-bar linkages and gear pairs, extrude outlines from images, guess shapes from photos, and size a hexacopter frame or a robot arm with a stand. I check printability, and I can control Blender with the keyboard and mouse."),
 ("what cant quadmesh do", ["what can't you do", "what are your limitations", "what is not supported"],
  "I cannot certify strength (my strength numbers are hand-calculation estimates), design complex organic or artistic models from scratch, or guess hidden geometry reliably from a photo. Always do the first real print supervised."),
 ("units dimensions", ["what units do you use", "how do I give dimensions", "are sizes in millimetres"],
  "Everything is in millimetres. Give sizes with mm, for example: a 60x40x5 mm plate with four 4.2 mm holes 8 mm from the edges."),
 ("printable meaning", ["what does printable mean", "does printable mean strong", "is a printable design safe"],
  "Printable means the geometry passed checks: watertight, wall thickness, overhang, fits the bed, correct size. It does not mean strong, safe or certified."),
 ("safety factor", ["what is a safety factor", "what safety factor should I use", "what does industrial grade mean"],
  "A safety factor is allowable stress divided into the load. Quadmesh uses 2 for hobby, 3 for industrial and 5 for critical parts, on derated printed-plastic strengths."),
 ("infill", ["how much infill do I need", "what infill for strong parts", "does infill matter"],
  "For strength, more perimeters matter more than infill. Use 3 or more walls and 30 to 50 percent infill; go higher only where screws or loads concentrate."),
 ("bed adhesion warp", ["why is my print warping", "how to stop warping", "print lifting from the bed"],
  "Warping comes from uneven cooling. Use a clean, heated bed, a brim, an enclosure for ABS or nylon, and avoid large flat corners."),
 ("bolts screws plastic", ["can I screw into a printed part", "how do I mount bolts in plastic", "heat set inserts"],
  "Use heat-set brass inserts or a nut trap for repeated screwing. A bare screw into plastic wears out; make the boss wall at least 1.5 times the screw diameter."),
 ("four bar linkage", ["what is a four bar linkage", "when does a crank turn fully", "what is Grashof"],
  "A four-bar linkage has a crank, coupler, rocker and ground link. If shortest plus longest is at most the sum of the other two (Grashof), one link can turn fully. Keep the transmission angle above 30 degrees so it does not bind."),
 ("hexacopter frame", ["how do you size a hexacopter frame", "what is thrust to weight", "how are the drone arms sized"],
  "I take the takeoff mass, assume 2:1 thrust-to-weight with a dynamic factor, and size each arm as a cantilever so its bending stress stays under the derated allowable divided by the safety factor."),
 ("robot arm sizing", ["how do you size a robot arm", "how is the base sized", "what loads does the stand carry"],
  "Each link is a cantilever loaded by the payload and the links beyond it; the base bolts see the overturning moment, and the base plate is sized against bolt pull-through."),
 ("speech recognition", ["how good is your speech recognition", "can you hear me", "why did you mishear me"],
  "I transcribe with my own speech model plus a vocabulary of my commands. When I am unsure I repeat what I heard and ask you to confirm before acting."),
 ("voice", ["can you talk", "what is your voice", "can you speak"],
  "I can speak my answers aloud with my own voice model when it is trained, or with your system voice until then."),
]
_STOP = set("a an the is are to of in it do does what how why can i you my me for on and or be this that with".split())


def _tok(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 1}


def kb_answer(question: str, threshold: float = 0.34) -> Optional[str]:
    q = _tok(question)
    if not q:
        return None
    best, best_s = None, 0.0
    for kw, qs, ans in FACTS:
        keys = _tok(kw) | set().union(*[_tok(x) for x in qs])
        core = _tok(kw)
        s = (len(q & keys) / len(q)) * (1.0 if q & core else 0.4)
        if s > best_s:
            best, best_s = ans, s
    return best if best_s >= threshold else None
