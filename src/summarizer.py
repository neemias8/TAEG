"""
Summarizer module using LEXRANK algorithm for text summarization.
"""

import nltk
from lexrank import LexRank
from typing import List, Dict, Any, Tuple
import numpy as np
from data_loader import ChronologyLoader, BiblicalDataLoader
import sys
from pathlib import Path
# Add parent directory to path to import improved_graph_builder
sys.path.insert(0, str(Path(__file__).parent.parent))
from improved_graph_builder import ImprovedTemporalGraphBuilder


class LexRankSummarizer:
    """Text summarizer using LEXRANK algorithm."""

    def __init__(self):
        """Initialize the LEXRANK summarizer."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        # Initialize LexRank without documents - will be set when summarizing
        self.lexrank = None

    def summarize_texts(self, texts: List[str], summary_length: int = 500) -> str:
        """
        Create a summary from multiple texts using LEXRANK for multi-document summarization.

        Args:
            texts: List of texts to summarize (each text represents a document)
            summary_length: Number of sentences in the summary

        Returns:
            Summarized text
        """
        if not texts:
            return ""

        # Method: Multi-document LEXRANK approach
        # Split each document into sentences and keep document boundaries
        document_sentences = []
        for text in texts:
            sentences = nltk.sent_tokenize(text)
            document_sentences.append(sentences)

        # Flatten all sentences for LEXRANK processing
        all_sentences = []
        for doc_sentences in document_sentences:
            all_sentences.extend(doc_sentences)

        if not all_sentences:
            return ""

        # Create LexRank instance with all sentences from all documents
        # This allows LEXRANK to find cross-document relationships
        lexrank = LexRank(all_sentences)

        # Get the most important sentences across all documents
        summary_sentences = lexrank.get_summary(all_sentences, summary_size=summary_length)

        # Join sentences into summary
        summary = ' '.join(summary_sentences)

        return summary

    def summarize_single_text(self, text: str, summary_length: int = 500) -> str:
        """
        Create a summary from a single text using LEXRANK.

        Args:
            text: Text to summarize
            summary_length: Number of sentences in the summary

        Returns:
            Summarized text
        """
        sentences = nltk.sent_tokenize(text)

        if not sentences:
            return ""

        # Create LexRank instance for these sentences
        lexrank = LexRank(sentences)

        # Get the most important sentences
        summary_sentences = lexrank.get_summary(sentences, summary_size=min(summary_length, len(sentences)))

        # Join sentences into summary
        summary = ' '.join(summary_sentences)

        return summary

    def get_sentence_scores(self, text: str) -> Dict[str, float]:
        """
        Get LEXRANK scores for each sentence in the text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping sentences to their scores
        """
        sentences = nltk.sent_tokenize(text)

        if not sentences:
            return {}

        # Create LexRank instance for these sentences
        lexrank = LexRank(sentences)

        # Get scores for all sentences
        scores = {}
        for sentence in sentences:
            # The LexRank library doesn't provide direct score access
            # We'll use a simple approach based on sentence centrality
            scores[sentence] = 1.0  # Placeholder - would need to implement proper scoring

        return scores


class LexRankTemporalAnchoring:
    """Text summarizer using LEXRANK with Temporal Anchoring (TA) based on biblical chronology."""

    def __init__(self):
        """Initialize the LEXRANK-TA summarizer."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        # Initialize loaders
        self.chrono_loader = ChronologyLoader()
        self.biblical_loader = BiblicalDataLoader()

        # Cache for gospel texts
        self._gospel_texts = None

    def _load_gospel_texts(self) -> Dict[str, str]:
        """Load and cache gospel texts."""
        if self._gospel_texts is None:
            self._gospel_texts = self.biblical_loader.load_all_gospels()
        return self._gospel_texts

    def _extract_event_text(self, event: Dict) -> List[str]:
        """
        Extract text segments for a specific event from all gospels that mention it.

        Args:
            event: Event dictionary from chronology

        Returns:
            List of text segments from different gospels
        """
        gospel_texts = self._load_gospel_texts()
        event_texts = []

        # Check each gospel for references to this event
        gospels = ['matthew', 'mark', 'luke', 'john']

        for gospel_key in gospels:
            if event.get(gospel_key):
                # Get the reference (e.g., "26:6-13")
                reference = event[gospel_key]

                # For simplicity, we'll use the entire chapter mentioned
                # In a more sophisticated implementation, we'd extract exact verses
                try:
                    # Extract chapter number from reference
                    chapter_part = reference.split(':')[0]
                    chapter_num = int(chapter_part)

                    # Get gospel name
                    gospel_name = gospel_key.capitalize()

                    # Load the specific chapter text
                    chapter_text = self._extract_chapter_text(gospel_name, chapter_num)

                    if chapter_text:
                        event_texts.append(chapter_text)

                except (ValueError, IndexError):
                    # If parsing fails, skip this reference
                    continue

        return event_texts

    def _extract_chapter_text(self, gospel: str, chapter_num: int) -> str:
        """
        Extract text for a specific chapter from a gospel.

        Args:
            gospel: Gospel name ('Matthew', 'Mark', 'Luke', 'John')
            chapter_num: Chapter number

        Returns:
            Chapter text or empty string if not found
        """
        try:
            # Load the specific gospel XML
            import xml.etree.ElementTree as ET
            from pathlib import Path

            gospels_map = {
                'Matthew': 'EnglishNIVMatthew40_PW.xml',
                'Mark': 'EnglishNIVMark41_PW.xml',
                'Luke': 'EnglishNIVLuke42_PW.xml',
                'John': 'EnglishNIVJohn43_PW.xml'
            }

            if gospel not in gospels_map:
                return ""

            xml_file = Path("data") / gospels_map[gospel]
            if not xml_file.exists():
                return ""

            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Find the specific chapter
            for chapter in root.findall('.//chapter'):
                if chapter.get('number') and int(chapter.get('number')) == chapter_num:
                    # Extract all verse texts from this chapter
                    verses = chapter.findall('.//verse')
                    chapter_text = ' '.join([verse.text for verse in verses if verse.text])
                    return chapter_text

        except Exception:
            return ""

        return ""

    def build_temporal_graph(self) -> Dict[str, Any]:
        """
        Build temporal graph from chronology XML with proper SAME_EVENT and BEFORE edges.

        Returns:
            Dictionary representing the temporal graph
        """
        events = self.chrono_loader.load_chronology()

        graph = {
            'nodes': {},
            'edges': []
        }

        # Create nodes for each event
        for event in events:
            event_id = event['id']
            graph['nodes'][f"event_{event_id}"] = {
                'event': event,
                'texts': self._extract_event_text(event)
            }

        # Create BEFORE edges between consecutive events
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]

            graph['edges'].append({
                'from': f"event_{current_event['id']}",
                'to': f"event_{next_event['id']}",
                'type': 'BEFORE'
            })

        # Create SAME_EVENT edges for events mentioned in multiple gospels
        same_event_pairs = self._find_same_event_pairs(events)

        for pair in same_event_pairs:
            event1_id = f"event_{pair[0]}"
            event2_id = f"event_{pair[1]}"

            # Only add if both nodes exist
            if event1_id in graph['nodes'] and event2_id in graph['nodes']:
                graph['edges'].append({
                    'from': event1_id,
                    'to': event2_id,
                    'type': 'SAME_EVENT'
                })

        return graph

    def _find_same_event_pairs(self, events: List[Dict]) -> List[Tuple[int, int]]:
        """
        Find pairs of events that represent the same chronological event
        mentioned in different gospels.

        Strategy: Look for events that are very close chronologically and have
        complementary gospel coverage (different gospels mentioning them).
        """
        pairs = []

        # Group events by chronological proximity (within 3 events)
        for i, event1 in enumerate(events):
            for j in range(i + 1, min(i + 4, len(events))):  # Check next 3 events
                event2 = events[j]

                # Get gospels mentioning each event
                gospels1 = set(g for g in ['matthew', 'mark', 'luke', 'john']
                              if event1.get(g) and event1[g].strip())
                gospels2 = set(g for g in ['matthew', 'mark', 'luke', 'john']
                              if event2.get(g) and event2[g].strip())

                # If they have complementary gospel coverage (some overlap or different gospels)
                if gospels1 and gospels2 and (gospels1 & gospels2 or gospels1 != gospels2):
                    # Check if descriptions are reasonably similar or if they're very close chronologically
                    if (self._descriptions_similar(event1['description'], event2['description'], threshold=0.3) or
                        abs(event1['id'] - event2['id']) <= 2):

                        pairs.append((event1['id'], event2['id']))

        return pairs

    def _descriptions_similar(self, desc1: str, desc2: str, threshold: float = 0.6) -> bool:
        """Check if two descriptions are similar using simple text overlap."""
        if not desc1 or not desc2:
            return False

        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union)
        return similarity >= threshold

    def summarize_with_temporal_anchoring(self, summary_length_per_event: int = 3, use_best_gospel: bool = False) -> str:
        """
        Generate summary using temporal anchoring approach with gospel-specific nodes.

        Args:
            summary_length_per_event: Number of sentences per event (ignored if use_best_gospel=True)
            use_best_gospel: If True, select the best gospel for each event instead of summarizing

        Returns:
            Concatenated summary of all events in chronological order
        """
        # Build the improved temporal graph with gospel-specific nodes
        graph_builder = ImprovedTemporalGraphBuilder()
        graph = graph_builder.build_improved_temporal_graph()

        # Group nodes by event_id for chronological processing
        event_nodes = {}
        for node_id, node_data in graph['nodes'].items():
            event_id = node_data['event_id']
            if event_id not in event_nodes:
                event_nodes[event_id] = []
            event_nodes[event_id].append((node_id, node_data))

        # Generate summary for each event in chronological order
        summaries = []
        events = self.chrono_loader.load_chronology()

        for event in events:
            event_id = event['id']

            if event_id in event_nodes:
                # Get all gospel versions for this event
                gospel_versions = event_nodes[event_id]
                event_texts = [(node_id, node_data) for node_id, node_data in gospel_versions if node_data['text']]

                if event_texts:
                    if use_best_gospel:
                        # Select the best gospel (longest text) for this event
                        best_node_id, best_node_data = max(event_texts, key=lambda x: len(x[1]['text']))
                        event_summary = best_node_data['text']
                        gospel_name = best_node_data['gospel'].capitalize()
                        print(f"📝 Event {event_id} ({event['description']}): Using complete text from {gospel_name} ({len(event_summary)} chars)")
                    else:
                        # Use multi-document LexRank if multiple gospels describe this event
                        texts_only = [node_data['text'] for node_id, node_data in event_texts]
                        if len(texts_only) > 1:
                            event_summary = self._summarize_multi_doc(texts_only, summary_length_per_event)
                            print(f"📝 Event {event_id} ({event['description']}): Multi-doc summary from {len(texts_only)} gospels")
                        else:
                            # Single document summary
                            event_summary = self._summarize_single_doc(texts_only[0], summary_length_per_event)
                            gospel_name = event_texts[0][1]['gospel'].capitalize()
                            print(f"📝 Event {event_id} ({event['description']}): Single-doc summary from {gospel_name}")

                    summaries.append(event_summary)
                else:
                    # If no text found, use event description as fallback
                    print(f"⚠️ Event {event_id} ({event['description']}): No text found, using description")
                    summaries.append(f"{event['description']}.")
            else:
                print(f"⚠️ Event {event_id} ({event['description']}): No nodes found")
                summaries.append(f"{event['description']}.")

        # Concatenate all event summaries
        full_summary = ' '.join(summaries)

        print(f"\n✅ Generated summary with {len(summaries)} event summaries")
        return full_summary

    def _summarize_multi_doc(self, texts: List[str], summary_length: int) -> str:
        """Summarize multiple documents using LexRank with improved multi-document handling."""
        if not texts:
            return ""

        # For biblical multi-document summarization, we want to:
        # 1. Extract sentences from all gospel versions
        # 2. Use LexRank to find the most representative sentences across versions
        # 3. Ensure we capture complementary information from different perspectives

        all_sentences = []
        sentence_sources = []  # Track which gospel each sentence comes from

        for i, text in enumerate(texts):
            sentences = nltk.sent_tokenize(text)
            all_sentences.extend(sentences)
            sentence_sources.extend([f"gospel_{i+1}"] * len(sentences))

        if not all_sentences:
            return ""

        # Create LexRank instance with all sentences from all gospel versions
        lexrank = LexRank(all_sentences)

        # Get summary sentences - allow more sentences since we have multiple perspectives
        target_sentences = min(summary_length * 2, len(all_sentences))  # Allow more sentences for multi-doc
        summary_sentences = lexrank.get_summary(all_sentences, summary_size=target_sentences)

        # If we got too many sentences, trim to the requested length
        if len(summary_sentences) > summary_length:
            summary_sentences = summary_sentences[:summary_length]

        return ' '.join(summary_sentences)

    def _summarize_single_doc(self, text: str, summary_length: int) -> str:
        """Summarize single document using LexRank."""
        sentences = nltk.sent_tokenize(text)

        if not sentences:
            return ""

        # Create LexRank instance
        lexrank = LexRank(sentences)

        # Get summary sentences
        summary_sentences = lexrank.get_summary(sentences, summary_size=min(summary_length, len(sentences)))

        return ' '.join(summary_sentences)
