"""
Data loader module for TAEG project.

This module handles loading and parsing of the five XML files:
- ChronologyOfTheFourGospels_PW.xml: Event structure and references
- EnglishNIVMatthew40_PW.xml: Gospel of Matthew
- EnglishNIVMark41_PW.xml: Gospel of Mark
- EnglishNIVLuke42_PW.xml: Gospel of Luke
- EnglishNIVJohn43_PW.xml: Gospel of John

Author: Your Name
Date: September 2025
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re
import xml.etree.ElementTree as ET

# Module-level logger (configured by the application)
logger = logging.getLogger(__name__)


@dataclass
class VerseReference:
    """Represents a verse span (e.g., "2:1-12" or "3:16-4:2")."""

    start_chapter: int
    start_verse: int
    end_chapter: Optional[int] = None
    end_verse: Optional[int] = None
    raw: str = ""

    _VERSE_PATTERN = re.compile(r"\s*(\d+)\s*:\s*([\dA-Za-z]+)\s*")

    @classmethod
    def from_string(cls, ref_str: str) -> "VerseReference":
        """Parse a verse reference string into a VerseReference."""
        if not ref_str or not ref_str.strip():
            raise ValueError("Empty verse reference")

        cleaned = ref_str.strip()
        parts = [p.strip() for p in cleaned.split('-') if p.strip()]
        if not parts:
            raise ValueError(f"Invalid verse reference format: {ref_str}")

        start_chapter, start_verse = cls._parse_chapter_verse(parts[0])

        if len(parts) == 1:
            return cls(
                start_chapter=start_chapter,
                start_verse=start_verse,
                end_chapter=start_chapter,
                end_verse=start_verse,
                raw=cleaned,
            )

        if len(parts) > 2:
            logger.warning(
                "Verse reference '%s' contains multiple hyphens; using first and last segments.",
                cleaned,
            )
        end_part = parts[-1]

        if ':' in end_part:
            end_chapter, end_verse = cls._parse_chapter_verse(end_part)
        else:
            end_chapter = start_chapter
            end_verse = cls._parse_verse_number(end_part)

        return cls(
            start_chapter=start_chapter,
            start_verse=start_verse,
            end_chapter=end_chapter,
            end_verse=end_verse,
            raw=cleaned,
        )

    @classmethod
    def _parse_chapter_verse(cls, token: str) -> Tuple[int, int]:
        match = cls._VERSE_PATTERN.fullmatch(token)
        if not match:
            raise ValueError(f"Invalid chapter:verse token: {token}")
        chapter = int(match.group(1))
        verse = cls._parse_verse_number(match.group(2))
        return chapter, verse

    @classmethod
    def _parse_verse_number(cls, token: str) -> int:
        match = re.match(r"(\d+)", token.strip())
        if not match:
            raise ValueError(f"Invalid verse token: {token}")
        return int(match.group(1))

    def __str__(self) -> str:
        end_chapter = self.end_chapter or self.start_chapter
        end_verse = self.end_verse if self.end_verse is not None else self.start_verse
        if end_chapter == self.start_chapter and end_verse == self.start_verse:
            return f"{self.start_chapter}:{self.start_verse}"
        if end_chapter == self.start_chapter:
            return f"{self.start_chapter}:{self.start_verse}-{end_verse}"
        return f"{self.start_chapter}:{self.start_verse}-{end_chapter}:{end_verse}"


@dataclass
class Event:
    """Represents a single event from the chronology."""

    event_id: str
    matthew_refs: List[VerseReference]
    mark_refs: List[VerseReference]
    luke_refs: List[VerseReference]
    john_refs: List[VerseReference]

    def get_all_gospels_with_refs(self) -> List[str]:
        """Return the list of gospels that include this event."""
        gospels = []
        if self.matthew_refs:
            gospels.append("matthew")
        if self.mark_refs:
            gospels.append("mark")
        if self.luke_refs:
            gospels.append("luke")
        if self.john_refs:
            gospels.append("john")
        return gospels


@dataclass
class Verse:
    """Represents a single verse with its content."""

    gospel: str
    chapter: int
    verse: int
    text: str


class DataLoader:
    """Main data loader class for the TAEG project."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.gospels = ["matthew", "mark", "luke", "john"]
        self.gospel_files = {
            "matthew": "EnglishNIVMatthew40_PW.xml",
            "mark": "EnglishNIVMark41_PW.xml",
            "luke": "EnglishNIVLuke42_PW.xml",
            "john": "EnglishNIVJohn43_PW.xml",
        }
        self.chronology_file = "ChronologyOfTheFourGospels_PW.xml"

        # In-memory storage
        self.events: List[Event] = []
        self.gospel_texts: Dict[str, Dict[int, Dict[int, str]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_all_data(self) -> Tuple[List[Event], Dict[str, Dict[int, Dict[int, str]]]]:
        """Load gospel texts and chronology events."""
        logger.info("Loading all XML data from %s", self.data_dir.resolve())

        self._load_gospel_texts()
        self._load_chronology()

        logger.info("Loaded %d events across %d gospels", len(self.events), len(self.gospel_texts))
        return self.events, self.gospel_texts

    # ------------------------------------------------------------------
    # Gospel parsing helpers
    # ------------------------------------------------------------------
    def _load_gospel_texts(self) -> None:
        for gospel, filename in self.gospel_files.items():
            filepath = self.data_dir / filename
            if not filepath.exists():
                logger.warning("Gospel file not found: %s", filepath)
                continue
            logger.info("Loading %s from %s", gospel, filepath.name)
            self.gospel_texts[gospel] = self._parse_gospel_xml(filepath)

    def _parse_gospel_xml(self, filepath: Path) -> Dict[int, Dict[int, str]]:
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError as exc:
            logger.error("XML parsing error for %s: %s", filepath, exc)
            return {}

        gospel_dict: Dict[int, Dict[int, str]] = {}

        for chapter_elem in root.findall('.//chapter'):
            chapter_attr = chapter_elem.get('number')
            if not chapter_attr or not chapter_attr.isdigit():
                continue
            chapter_num = int(chapter_attr)
            gospel_dict.setdefault(chapter_num, {})

            for verse_elem in chapter_elem.findall('.//verse'):
                verse_attr = verse_elem.get('number')
                if not verse_attr or not verse_attr.isdigit():
                    continue
                verse_num = int(verse_attr)
                verse_text = (verse_elem.text or "").strip()
                gospel_dict[chapter_num][verse_num] = verse_text

        return gospel_dict

    # ------------------------------------------------------------------
    # Chronology parsing helpers
    # ------------------------------------------------------------------
    def _load_chronology(self) -> None:
        filepath = self.data_dir / self.chronology_file
        if not filepath.exists():
            logger.error("Chronology file not found: %s", filepath)
            return

        logger.info("Loading chronology from %s", filepath.name)

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError as exc:
            logger.error("XML parsing error for chronology: %s", exc)
            return

        for event_elem in root.findall('.//event'):
            event_id = event_elem.get('id', '').strip()
            if not event_id:
                logger.warning("Encountered chronology event without id attribute; skipping")
                continue

            matthew_refs = self._parse_verse_references(event_elem.findall('matthew'))
            mark_refs = self._parse_verse_references(event_elem.findall('mark'))
            luke_refs = self._parse_verse_references(event_elem.findall('luke'))
            john_refs = self._parse_verse_references(event_elem.findall('john'))

            self.events.append(
                Event(
                    event_id=event_id,
                    matthew_refs=matthew_refs,
                    mark_refs=mark_refs,
                    luke_refs=luke_refs,
                    john_refs=john_refs,
                )
            )

    def _parse_verse_references(self, elements: Iterable[ET.Element]) -> List[VerseReference]:
        references: List[VerseReference] = []

        for elem in elements:
            if elem is None:
                continue
            refs_text = (elem.text or '').strip()
            if not refs_text:
                continue

            for ref_str in re.split(r'[;,]', refs_text):
                ref_str = ref_str.strip()
                if not ref_str:
                    continue
                try:
                    references.append(VerseReference.from_string(ref_str))
                except ValueError as exc:
                    logger.warning("Could not parse verse reference '%s': %s", ref_str, exc)

        return references

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_verses_for_event(self, event: Event) -> Dict[str, List[Verse]]:
        """Extract verse texts for a given event across the gospels."""
        result: Dict[str, List[Verse]] = {}

        gospel_refs = {
            'matthew': event.matthew_refs,
            'mark': event.mark_refs,
            'luke': event.luke_refs,
            'john': event.john_refs,
        }

        for gospel, refs in gospel_refs.items():
            if not refs or gospel not in self.gospel_texts:
                result[gospel] = []
                continue

            verses: List[Verse] = []
            for ref in refs:
                resolved, _, _ = self._expand_reference_indices(gospel, ref)
                gospel_text = self.gospel_texts[gospel]
                for chapter, verse_num in resolved:
                    text = gospel_text.get(chapter, {}).get(verse_num)
                    if text is None:
                        continue
                    verses.append(
                        Verse(
                            gospel=gospel,
                            chapter=chapter,
                            verse=verse_num,
                            text=text,
                        )
                    )
            result[gospel] = verses

        return result

    def get_concatenated_text_for_event(self, event: Event) -> str:
        verses_dict = self.get_verses_for_event(event)
        snippets: List[str] = []
        for gospel in self.gospels:
            for verse in verses_dict.get(gospel, []):
                snippets.append(f"[{gospel.title()} {verse.chapter}:{verse.verse}] {verse.text}")
        return " ".join(snippets)

    def get_event_sequence_by_gospel(self, gospel: str) -> List[str]:
        if gospel not in self.gospels:
            raise ValueError(f"Invalid gospel: {gospel}")
        sequence: List[str] = []
        for event in self.events:
            refs = getattr(event, f"{gospel}_refs")
            if refs:
                sequence.append(event.event_id)
        return sequence

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def validate_data(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            'total_events': len(self.events),
            'gospels_loaded': list(self.gospel_texts.keys()),
            'events_per_gospel': {},
            'total_verses_per_gospel': {},
            'missing_references': [],
            'validation_errors': [],
        }

        for gospel in self.gospels:
            stats['events_per_gospel'][gospel] = sum(
                1 for event in self.events if getattr(event, f"{gospel}_refs")
            )

        for gospel, chapters in self.gospel_texts.items():
            stats['total_verses_per_gospel'][gospel] = sum(len(verses) for verses in chapters.values())

        for event in self.events:
            for gospel in self.gospels:
                refs = getattr(event, f"{gospel}_refs")
                if not refs:
                    continue

                if gospel not in self.gospel_texts:
                    stats['validation_errors'].append(
                        f"Gospel {gospel} not loaded but referenced by event {event.event_id}"
                    )
                    continue

                for ref in refs:
                    _, missing_verses, missing_chapters = self._expand_reference_indices(gospel, ref)
                    for chapter in missing_chapters:
                        stats['missing_references'].append(
                            f"Event {event.event_id}: Missing chapter {chapter} in {gospel} (reference {ref.raw or ref})"
                        )
                    for chapter, verse in missing_verses:
                        stats['missing_references'].append(
                            f"Event {event.event_id}: Missing verse {chapter}:{verse} in {gospel} (reference {ref.raw or ref})"
                        )

        return stats

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _expand_reference_indices(
        self,
        gospel: str,
        ref: VerseReference,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[int]]:
        """Resolve a reference into concrete verse indices.

        Returns a tuple of (resolved, missing_verses, missing_chapters).
        """
        resolved: List[Tuple[int, int]] = []
        missing_verses: List[Tuple[int, int]] = []
        missing_chapters: List[int] = []

        gospel_text = self.gospel_texts.get(gospel)
        if not gospel_text:
            missing_chapters.append(ref.start_chapter)
            return resolved, missing_verses, missing_chapters

        start_chapter = ref.start_chapter
        end_chapter = ref.end_chapter or ref.start_chapter
        end_chapter = max(start_chapter, end_chapter)

        for chapter in range(start_chapter, end_chapter + 1):
            chapter_text = gospel_text.get(chapter)
            if not chapter_text:
                missing_chapters.append(chapter)
                continue

            verses_in_chapter = sorted(chapter_text.keys())
            if not verses_in_chapter:
                missing_chapters.append(chapter)
                continue

            start_verse = ref.start_verse if chapter == start_chapter else verses_in_chapter[0]
            if chapter == end_chapter:
                if ref.end_verse is not None:
                    end_verse = ref.end_verse
                elif chapter == start_chapter:
                    end_verse = ref.start_verse
                else:
                    end_verse = verses_in_chapter[-1]
            else:
                end_verse = verses_in_chapter[-1]

            if end_verse < start_verse:
                start_verse, end_verse = end_verse, start_verse

            for verse_num in range(start_verse, end_verse + 1):
                if verse_num in chapter_text:
                    resolved.append((chapter, verse_num))
                else:
                    missing_verses.append((chapter, verse_num))

        return resolved, missing_verses, missing_chapters


def main() -> None:
    """Example usage of the DataLoader."""
    loader = DataLoader("data")
    events, _ = loader.load_all_data()

    stats = loader.validate_data()
    print("Data validation results:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if events:
        first_event = events[0]
        print(f"\nFirst event (ID: {first_event.event_id}):")
        text = loader.get_concatenated_text_for_event(first_event)
        print(f"Text: {text[:200]}...")
        matthew_sequence = loader.get_event_sequence_by_gospel('matthew')
        print(f"\nMatthew sequence (first 10 events): {matthew_sequence[:10]}")


if __name__ == "__main__":
    main()
