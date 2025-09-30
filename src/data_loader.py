"""
Data loader module for processing biblical texts from XML files.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from pathlib import Path


class BiblicalDataLoader:
    """Loader for biblical text data from XML files."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing the XML files
        """
        self.data_dir = Path(data_dir)
        self.gospels = {
            'Matthew': 'EnglishNIVMatthew40_PW.xml',
            'Mark': 'EnglishNIVMark41_PW.xml',
            'Luke': 'EnglishNIVLuke42_PW.xml',
            'John': 'EnglishNIVJohn43_PW.xml'
        }

        # Chapters that contain the Passion Week narrative
        self.passion_week_chapters = {
            'Matthew': list(range(21, 29)),  # Chapters 21-28
            'Mark': list(range(11, 17)),     # Chapters 11-16
            'Luke': list(range(19, 25)),     # Chapters 19-24
            'John': [12, 13, 14, 15, 16, 17, 18, 19, 20]  # Chapters 12-20
        }

    def load_gospel_text(self, gospel: str) -> str:
        """
        Load the passion week text from a specific gospel.

        Args:
            gospel: Name of the gospel ('Matthew', 'Mark', 'Luke', or 'John')

        Returns:
            Concatenated text from all verses in the passion week chapters
        """
        if gospel not in self.gospels:
            raise ValueError(f"Unknown gospel: {gospel}")

        xml_file = self.data_dir / self.gospels[gospel]
        if not xml_file.exists():
            raise FileNotFoundError(f"XML file not found: {xml_file}")

        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        verses = []

        # Find chapters that belong to passion week
        target_chapters = self.passion_week_chapters[gospel]

        for chapter in root.findall(".//chapter"):
            chapter_num = int(chapter.get('number', 0))
            if chapter_num in target_chapters:
                # Extract all verses from this chapter
                for verse in chapter.findall("verse"):
                    verse_text = verse.text
                    if verse_text:
                        verses.append(verse_text.strip())

        return ' '.join(verses)

    def load_all_gospels(self) -> Dict[str, str]:
        """
        Load passion week texts from all four gospels.

        Returns:
            Dictionary mapping gospel names to their texts
        """
        return {gospel: self.load_gospel_text(gospel) for gospel in self.gospels.keys()}

    def load_golden_sample(self) -> str:
        """
        Load the golden sample text.

        Returns:
            The golden sample text
        """
        golden_file = self.data_dir / "Golden_Sample.txt"
        if not golden_file.exists():
            raise FileNotFoundError(f"Golden sample file not found: {golden_file}")

        with open(golden_file, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def get_sentences_from_text(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting - can be improved with NLTK
        sentences = []
        current_sentence = []

        for word in text.split():
            current_sentence.append(word)
            if word.endswith('.') or word.endswith('!') or word.endswith('?'):
                sentences.append(' '.join(current_sentence))
                current_sentence = []

        if current_sentence:
            sentences.append(' '.join(current_sentence))

        return sentences


class ChronologyLoader:
    """Loader for biblical chronology data from XML files."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the chronology loader.

        Args:
            data_dir: Directory containing the chronology XML file
        """
        self.data_dir = Path(data_dir)
        self.chronology_file = self.data_dir / "ChronologyOfTheFourGospels_PW.xml"

    def load_chronology(self) -> List[Dict]:
        """
        Load the chronological events from the XML file.

        Returns:
            List of event dictionaries with chronological order
        """
        if not self.chronology_file.exists():
            raise FileNotFoundError(f"Chronology file not found: {self.chronology_file}")

        tree = ET.parse(self.chronology_file)
        root = tree.getroot()

        events = []
        for event_elem in root.find('events'):
            event = {
                'id': int(event_elem.get('id')),
                'day': event_elem.find('day').text if event_elem.find('day') is not None else '',
                'description': event_elem.find('description').text if event_elem.find('description') is not None else '',
                'when_where': event_elem.find('when_where').text if event_elem.find('when_where') is not None else '',
                'matthew': event_elem.find('matthew').text if event_elem.find('matthew') is not None else '',
                'mark': event_elem.find('mark').text if event_elem.find('mark') is not None else '',
                'luke': event_elem.find('luke').text if event_elem.find('luke') is not None else '',
                'john': event_elem.find('john').text if event_elem.find('john') is not None else ''
            }
            events.append(event)

        # Sort by ID to ensure chronological order
        events.sort(key=lambda x: x['id'])
        return events

    def get_event_descriptions(self) -> List[str]:
        """
        Get list of event descriptions in chronological order.

        Returns:
            List of event descriptions
        """
        events = self.load_chronology()
        return [event['description'] for event in events]

    def get_event_ids(self) -> List[int]:
        """
        Get list of event IDs in chronological order.

        Returns:
            List of event IDs
        """
        events = self.load_chronology()
        return [event['id'] for event in events]