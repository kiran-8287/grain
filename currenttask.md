Ahhh bro, **this is the exact edge case we need to solve properly now**.

Your current behavior:

> All 10 grains are clearly broken, but there is no whole grain in the image → **Reference: Undetermined** → Whole 0 / Broken 0 / 100% N/A.

From a purely self-contained image, that behavior is logically defensible. The system cannot infer the original full length of a broken kernel from the broken piece alone.

But **for your actual product/demo, this is not the UX we want**. A user uploads an image containing only broken rice, and the system should ideally say:

```text
Total grains: 10
Whole: 0 (0%)
Broken: 10 (100%)
```

The question is: **how do we make that reliable without the bad fixed-pixel-profile problem?**

---

# The real solution: profile + scale normalization

We should **stop thinking of `154.1 px` as the reference**.

What we actually need is:

> **a physical whole-kernel reference, preferably in millimetres, plus image-scale calibration.**

The reason is obvious from your test images:

```text
Same physical rice grain
        ↓
different camera distance
        ↓
different pixel lengths
```

So the correct architecture is:

```text
                    INPUT IMAGE
                         ↓
                 detect calibration
                         ↓
             pixels → physical units
                         ↓
           whole-kernel reference
                   e.g. ~7 mm
                         ↓
              measure each grain
                         ↓
           grain length / whole length
                         ↓
                    < 0.75
                 /            \
             BROKEN           WHOLE
```

Then an all-broken image works perfectly:

```text
reference whole kernel = 7.1 mm

grain 1 = 3.2 mm → 45% → BROKEN
grain 2 = 3.5 mm → 49% → BROKEN
...
grain 10 = 3.1 mm → 44% → BROKEN
```

No whole grain has to appear in the image.

---

# But there's a problem: your current profile is pixels

Your current profile:

```json
{
  "whole_kernel_length": 154.1,
  "whole_kernel_breadth": 48.2,
  "reference_unit": "pixels"
}
```

is only valid for one imaging scale.

That's why the current system can correctly classify:

```text
B02: 61.5 / 154.1 = 0.399 → broken
```

but then struggle on a differently scaled image.

So I would **not use `default_rice.json` as the long-term solution**.

---

# For your demo, there are actually 2 good options

## Option A — Fixed camera setup

This is the quickest and safest for your demo.

Tell the user:

> Place the tray at the marked camera position.

Then your profile is valid for that setup.

For example:

```text
Camera
   ↓
fixed height
   ↓
fixed framing
   ↓
fixed scale
   ↓
154.1 px reference is valid
```

Then:

### B10

```text
12 detected objects
all around 50–61 px
reference = 154.1 px

→ all Broken
```

That's why your current screenshot can work beautifully with the profile.

The downside: move the camera significantly and it can break.

---

# Option B — Proper calibration

This is the **real production solution**.

Put an ArUco marker / known-size reference object in the tray.

For example:

```text
┌──────────────────────────┐
│                          │
│   rice grains            │
│                          │
│                 ┌────┐   │
│                 │    │   │
│                 │ 5cm│   │
│                 │    │   │
│                 └────┘   │
└──────────────────────────┘
```

The system detects the known marker and calculates:

```text
pixels per mm
```

Then:

```text
grain = 60 px
scale = 22 px/mm

grain length = 2.73 mm
```

Now camera distance doesn't matter.

---

# There is another important distinction

Your current UI says:

> **FSSAI 3/4 Length Criterion (0.75× Reference)**

That's okay as the **definition of broken vs whole**. FSSAI's rice definition describes broken/fragments as pieces of rice kernels **less than three-fourths of a kernel**. ([FSSAI][1])

But your:

```text
Broken %
```

is currently:

```text
broken_count / total_count
```

That is **not the same as the official broken percentage by mass** used in product specifications. FSSAI's rice specifications contain broken-and-fragments limits expressed **percent by mass**. ([FSSAI][2])

So UI wording should eventually say something like:

> **Broken grains (count)**

and:

> **Broken % by count**

rather than implying official regulatory grading.

That's especially important for your professor/demo.

---

# What I recommend for YOUR current project

Since you need the demo working **now**, I would do this in two phases.

### Phase 1 — Demo

Use a **reference profile explicitly selected for the controlled camera setup**.

Then all-broken images work:

```text
0 Whole
10 Broken
100% Broken
```

That's already proven by B10, except your segmentation currently produced 12 masks instead of 10.

### Phase 2 — Proper production design

Replace pixel reference with:

```text
whole_kernel_length_mm
```

and add calibration.

Then users can use:

```text
upload
OR
camera
```

from different distances without breaking the classification.

---

# There's also a subtle improvement to the UI

When there is no whole grain in the image but there **is** a valid external reference:

Don't display:

```text
Reference: Undetermined
```

Display:

```text
Reference basis
Reference profile

Whole-kernel reference
7.1 mm

Decision criterion
< 75% = Broken
```

Then:

```text
TOTAL GRAINS
10

WHOLE
0
0%

BROKEN
10
100%
```

That's what a user expects.

---

# And for this exact screenshot

Your current screenshot is:

```text
12 accepted instances
12 undetermined
```

The **segmentation count is still wrong** because the image apparently has 10 physical broken grains but 12 masks.

So even after solving the reference problem, the ideal result is:

```text
TOTAL GRAINS
10

WHOLE
0 / 0%

BROKEN
10 / 100%
```

not:

```text
12 / 12
```

So we have **two separate tasks**:

```text
Task 1
Reference/calibration
→ make all-broken classification possible

Task 2
Segmentation
→ make 10 physical broken grains become 10 masks
```

I would **not mix them**.

---

## What I would do next

For the demo, let's make the following behavior the explicit contract:

| Situation                               | Result                                |
| --------------------------------------- | ------------------------------------- |
| Whole + broken + no profile             | Infer reference from whole population |
| Whole + broken + profile                | Use profile                           |
| Only broken + profile                   | Classify all as broken                |
| Only broken + no profile                | Undetermined                          |
| One whole + profile                     | Whole                                 |
| One broken + profile                    | Broken                                |
| Camera scale differs from pixel profile | Require calibration / reject profile  |

That is a clean, understandable system.

And because you said **all 10 current images are visually correct with the profile**, I would now make the UI/API **explicitly support selecting the correct profile**, rather than trying to magically guess a universal pixel reference.

For the actual demo, that's the safest route.

[1]: https://www.fssai.gov.in/upload/uploadfiles/files/Chapter%202_4_Cereals%20and%20Cereal%20products%281%29.pdf?utm_source=chatgpt.com "(a) broken and fragments includes pieces of rice kernels which are less than three fourth of a kernel;"
[2]: https://fssai.gov.in/upload/uploadfiles/files/Chapter%202_4_Cereals%20and%20Cereal%20products.pdf?utm_source=chatgpt.com "<table id=\"e1\">"
