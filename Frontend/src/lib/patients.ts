/**
 * Display identities for studies.
 *
 * The corpus is de-identified, so the only handle a study has is its
 * StudyInstanceUID — sixty characters of digits and dots that no one can read,
 * compare at a glance, or say out loud. This module trades that for a
 * pseudonym: a name, a medical record number, and an age/sex line, all derived
 * from the UID itself.
 *
 * Derived, not stored, and that is the point. The same UID always produces the
 * same person, in every view and across reloads, with no table to keep in sync
 * and no round-trip to ask for one. None of it is real and none of it leaves
 * the browser — the UID stays the only identifier the API ever sees.
 */

import type { PatientIdentity } from "@/interfaces";

/**
 * FNV-1a, 32-bit, finished with Murmur3's avalanche step. Not a security hash:
 * the job is a well-spread number from a short string, and `Math.imul` is what
 * keeps each multiply in 32-bit range rather than drifting into float
 * territory.
 *
 * The avalanche is not optional here. Study UIDs share a long prefix and differ
 * only near the end, and every draw below reduces the hash modulo a small
 * number — that is, it reads the low bits. Raw FNV-1a barely moves those, so
 * thousands of studies collapse onto a handful of names. The three shift-xor
 * rounds fold the high bits down into the low ones, which is what makes the
 * whole 32-bit result count.
 */
function hash(seed: string): number {
  let value = 0x811c9dc5;
  for (let index = 0; index < seed.length; index += 1) {
    value ^= seed.charCodeAt(index);
    value = Math.imul(value, 0x01000193);
  }
  value ^= value >>> 16;
  value = Math.imul(value, 0x85ebca6b);
  value ^= value >>> 13;
  value = Math.imul(value, 0xc2b2ae35);
  value ^= value >>> 16;
  return value >>> 0;
}

/**
 * One draw from a UID. Every field salts the seed differently, so a study whose
 * hash lands on the first surname is not thereby pushed to the first given
 * name: the fields vary independently.
 */
function pick<T>(uid: string, field: string, pool: readonly T[]): T {
  return pool[hash(`${field}#${uid}`) % pool.length];
}

function number(uid: string, field: string, min: number, max: number): number {
  return min + (hash(`${field}#${uid}`) % (max - min + 1));
}

// Deliberately wide-ranging: a knee clinic's list is not all one background,
// and a demo list that reads as one is a worse demo.
// prettier-ignore
const GIVEN_NAMES_F = [
  "Amara", "Lena", "Sofia", "Priya", "Yuki", "Noor", "Clara", "Ines",
  "Mariam", "Elif", "Rosa", "Anika", "Chiara", "Freya", "Nadia", "Leila",
  "Camille", "Zoe", "Hannah", "Aisha", "Marta", "Ivy", "Tamar", "Saoirse",
  "Delphine", "Nina", "Ana", "Keiko", "Beatriz", "Ada", "Meera", "Astrid",
] as const;

// prettier-ignore
const GIVEN_NAMES_M = [
  "Julian", "Omar", "Mateo", "Arjun", "Kenji", "Samir", "Felix", "Tomas",
  "Idris", "Levi", "Hugo", "Rafael", "Nikolai", "Anders", "Karim", "Elias",
  "Theo", "Ravi", "Louis", "Amir", "Diego", "Oskar", "Jonas", "Cillian",
  "Malik", "Viktor", "Andres", "Haruto", "Emilio", "Ezra", "Dev", "Soren",
] as const;

// prettier-ignore
const FAMILY_NAMES = [
  "Okafor", "Lindqvist", "Moreau", "Delgado", "Nakamura", "Haddad", "Petrova",
  "Whitfield", "Oyelaran", "Bergström", "Fontaine", "Navarro", "Tanaka", "Aziz",
  "Novak", "Ashcroft", "Mensah", "Halvorsen", "Rousseau", "Castellano", "Kimura",
  "Rahimi", "Sokolov", "Barlow", "Diallo", "Solberg", "Lefevre", "Iglesias",
  "Watanabe", "Farouk", "Kovac", "Merrick", "Adeyemi", "Nilsson", "Marchand",
  "Vargas", "Fujimoto", "Nasser", "Marek", "Ellery", "Boateng", "Dahl",
  "Perrin", "Salazar", "Ishikawa", "Karam", "Dvorak", "Sinclair",
] as const;

/**
 * Avatar palettes. Each is a written-out pair of Tailwind classes rather than a
 * colour to interpolate, because Tailwind only ships the classes it can find
 * spelled out in the source.
 */
const TONES = [
  "bg-violet-100 text-violet-700",
  "bg-sky-100 text-sky-700",
  "bg-emerald-100 text-emerald-700",
  "bg-amber-100 text-amber-700",
  "bg-rose-100 text-rose-700",
  "bg-teal-100 text-teal-700",
  "bg-indigo-100 text-indigo-700",
  "bg-fuchsia-100 text-fuchsia-700",
] as const;

/** The pseudonym for a study UID. Pure and stable: same UID, same person. */
export function patientOf(uid: string): PatientIdentity {
  const sex = hash(`sex#${uid}`) % 2 === 0 ? "F" : "M";
  const given = pick(uid, "given", sex === "F" ? GIVEN_NAMES_F : GIVEN_NAMES_M);
  const family = pick(uid, "family", FAMILY_NAMES);

  return {
    name: `${given} ${family}`,
    initials: `${given[0]}${family[0]}`,
    // Grouped the way a wristband groups it — far easier to read back than
    // seven loose digits.
    mrn: `${number(uid, "mrn-a", 100, 999)}-${number(uid, "mrn-b", 1000, 9999)}`,
    age: number(uid, "age", 19, 84),
    sex,
    tone: pick(uid, "tone", TONES),
  };
}

/** True when a filter query matches anything about a study a viewer can see. */
export function matchesPatient(uid: string, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;

  const patient = patientOf(uid);
  return (
    patient.name.toLowerCase().includes(needle) ||
    patient.mrn.includes(needle) ||
    // The UID is off the screen now, but a pasted one should still find its
    // study — that is how a developer or a support ticket arrives at a row.
    uid.toLowerCase().includes(needle)
  );
}

/**
 * A short, readable handle for one acquisition run, e.g. `A4F2`.
 *
 * Series UIDs are as long as study UIDs and a study carries several of them, so
 * the list still needs something to tell two sagittal runs apart — just not a
 * line of unreadable digits under every card.
 */
export function seriesCode(seriesUid: string): string {
  return hash(`series#${seriesUid}`).toString(16).toUpperCase().padStart(8, "0").slice(0, 4);
}
