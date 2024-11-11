# Prompt Engineering


## Principles of Prompting

- Naive Prompt

```txt
Can I have a list of product names for a pair of shoes that fit any shoe size?
```

- Give direction

```txt
"Product description: A pair of shoes that can fit any foot size
Seed words: adaptable, fit, omni-fit
Product names:"
```

- Specify Format

```txt
"Product description: A home milkshake maker
Seed words: fast, healthy, compact
Product names: HomeShaker, Fit Shaker, QuickShake, Shake Maker

Product description: A pair of shoes that can fit any foot size
Seed words: adjustable, bigfoot, universal
Product names:"
```

- Provide Examples

```txt
"Brainstorm product names as a comma separated list.

Product description: A watch that can tell accurate time in space
Seed words: astronaut, space-hardened, eliptical orbit
Product names: iNaut, iSpace, iTime

Product description: A home milkshake maker
Seed words: fast, healthy, compact
Product names: iShake, iSmoothie, iShake Mini

Product description: A pair of shoes that can fit any foot size
Seed words: adaptable, fit, omni-fit
Product names:"
```

- Evaluate Quality

```txt
"Brainstorm product names as a comma separated list.

Product description: A refrigerator that dispenses beer
Seed words: beer, drink, bar
Product names: iBarFridge, iFridgeBeer, iDrinkBeerFridge

Product description: A watch that can tell accurate time in space
Seed words: astronaut, space-hardened, eliptical orbit
Product names: iNaut, iSpace, iTime

Product description: A home milkshake maker
Seed words: fast, healthy, compact
Product names: iShake, iSmoothie, iShake Mini

Product description: A pair of shoes that can fit any foot size
Seed words: adaptable, fit, omni-fit
Product names:"
```

- Divide Labor

```txt
"Brainstorm product names as a comma separated list.

Product description: A refrigerator that dispenses beer
Seed words: beer, drink, bar
Product names: iBarFridge, iFridgeBeer, iDrinkBeerFridge

Product description: A watch that can tell accurate time in space
Seed words: astronaut, space-hardened, eliptical orbit
Product names: iNaut, iSpace, iTime

Product description: A home milkshake maker
Seed words: fast, healthy, compact
Product names: iShake, iSmoothie, iShake Mini

Product description: A pair of shoes that can fit any foot size
Seed words: adaptable, fit, omni-fit
Product names:"

--------------------------------
Please rate the product names for "A pair of shoes that can fit any foot size" based on their catchiness, uniqueness, and simplicity. Rate them on a scale from 1-5, with 5 being the highest score. Respond only with a table containing the results.
```

## AI Hallucinations

```txt
<Question>
If you don't have a factual answer, just reply with 'Unable to reply'.
Avoid making up a story, and make sure that the ouput is reliable
```
