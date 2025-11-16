"""
English lesson content for the AI Tutor.
Provides spelling and word-finding tasks with hints and explanations.
"""
from typing import Dict, List, Optional


class EnglishLessons:
    """English lesson content organized by difficulty level."""
    
    def __init__(self):
        """Initialize English lessons database."""
        self.lessons = self._initialize_lessons()
    
    def _initialize_lessons(self) -> Dict[int, List[Dict]]:
        """
        Initialize lesson database with questions, answers, hints, and explanations.
        
        Returns:
            Dictionary mapping difficulty levels to lists of lesson items
        """
        lessons = {
            1: [  # Very Easy - 3-letter words
                {
                    "question": "Spell the word: C-A-T",
                    "answer": "cat",
                    "hint": "It's a furry pet that says 'meow'.",
                    "explanation": "C-A-T spells 'cat'. It's a small furry animal.",
                    "type": "spelling",
                    "word": "cat"
                },
                {
                    "question": "Spell the word: D-O-G",
                    "answer": "dog",
                    "hint": "It's a friendly pet that says 'woof'.",
                    "explanation": "D-O-G spells 'dog'. It's a loyal pet animal.",
                    "type": "spelling",
                    "word": "dog"
                },
                {
                    "question": "Spell the word: S-U-N",
                    "answer": "sun",
                    "hint": "It's bright and yellow in the sky during the day.",
                    "explanation": "S-U-N spells 'sun'. It gives us light and warmth.",
                    "type": "spelling",
                    "word": "sun"
                },
                {
                    "question": "Spell the word: H-A-T",
                    "answer": "hat",
                    "hint": "You wear it on your head.",
                    "explanation": "H-A-T spells 'hat'. It's something you wear on your head.",
                    "type": "spelling",
                    "word": "hat"
                },
                {
                    "question": "Spell the word: B-A-T",
                    "answer": "bat",
                    "hint": "It's used in baseball, or it's a flying animal.",
                    "explanation": "B-A-T spells 'bat'. It can be a sports tool or a flying animal.",
                    "type": "spelling",
                    "word": "bat"
                },
            ],
            2: [  # Easy - 4-letter words
                {
                    "question": "Spell the word: B-O-O-K",
                    "answer": "book",
                    "hint": "You read it. It has pages.",
                    "explanation": "B-O-O-K spells 'book'. It's something you read with pages.",
                    "type": "spelling",
                    "word": "book"
                },
                {
                    "question": "Spell the word: T-R-E-E",
                    "answer": "tree",
                    "hint": "It grows tall and has leaves.",
                    "explanation": "T-R-E-E spells 'tree'. It's a tall plant with branches and leaves.",
                    "type": "spelling",
                    "word": "tree"
                },
                {
                    "question": "Spell the word: F-I-S-H",
                    "answer": "fish",
                    "hint": "It lives in water and has fins.",
                    "explanation": "F-I-S-H spells 'fish'. It's an animal that lives in water.",
                    "type": "spelling",
                    "word": "fish"
                },
                {
                    "question": "Spell the word: B-I-R-D",
                    "answer": "bird",
                    "hint": "It has wings and can fly.",
                    "explanation": "B-I-R-D spells 'bird'. It's an animal with wings that can fly.",
                    "type": "spelling",
                    "word": "bird"
                },
                {
                    "question": "Spell the word: H-O-M-E",
                    "answer": "home",
                    "hint": "It's where you live.",
                    "explanation": "H-O-M-E spells 'home'. It's the place where you live.",
                    "type": "spelling",
                    "word": "home"
                },
            ],
            3: [  # Medium - 5-letter words and word finding
                {
                    "question": "Find the word that means 'happy': glad, sad, mad",
                    "answer": "glad",
                    "hint": "Which word means the same as 'happy'?",
                    "explanation": "'Glad' means the same as 'happy'. Both mean feeling good.",
                    "type": "word_finding",
                    "options": ["glad", "sad", "mad"]
                },
                {
                    "question": "Spell the word: A-P-P-L-E",
                    "answer": "apple",
                    "hint": "It's a red or green fruit.",
                    "explanation": "A-P-P-L-E spells 'apple'. It's a round fruit that can be red or green.",
                    "type": "spelling",
                    "word": "apple"
                },
                {
                    "question": "Find the word that means 'big': small, large, tiny",
                    "answer": "large",
                    "hint": "Which word means the same as 'big'?",
                    "explanation": "'Large' means the same as 'big'. Both mean something is not small.",
                    "type": "word_finding",
                    "options": ["small", "large", "tiny"]
                },
                {
                    "question": "Spell the word: H-O-U-S-E",
                    "answer": "house",
                    "hint": "It's a building where people live.",
                    "explanation": "H-O-U-S-E spells 'house'. It's a building where families live.",
                    "type": "spelling",
                    "word": "house"
                },
                {
                    "question": "Find the word that means 'fast': slow, quick, lazy",
                    "answer": "quick",
                    "hint": "Which word means the same as 'fast'?",
                    "explanation": "'Quick' means the same as 'fast'. Both mean moving or happening rapidly.",
                    "type": "word_finding",
                    "options": ["slow", "quick", "lazy"]
                },
            ],
            4: [  # Medium-Hard - 6-letter words and synonyms
                {
                    "question": "Spell the word: F-R-I-E-N-D",
                    "answer": "friend",
                    "hint": "It's someone you like and play with.",
                    "explanation": "F-R-I-E-N-D spells 'friend'. It's a person you like and spend time with.",
                    "type": "spelling",
                    "word": "friend"
                },
                {
                    "question": "Find the word that means 'smart': clever, silly, funny",
                    "answer": "clever",
                    "hint": "Which word means the same as 'smart'?",
                    "explanation": "'Clever' means the same as 'smart'. Both mean being intelligent.",
                    "type": "word_finding",
                    "options": ["clever", "silly", "funny"]
                },
                {
                    "question": "Spell the word: S-C-H-O-O-L",
                    "answer": "school",
                    "hint": "It's a place where you learn.",
                    "explanation": "S-C-H-O-O-L spells 'school'. It's a place where students go to learn.",
                    "type": "spelling",
                    "word": "school"
                },
                {
                    "question": "Find the word that means 'beautiful': pretty, ugly, old",
                    "answer": "pretty",
                    "hint": "Which word means the same as 'beautiful'?",
                    "explanation": "'Pretty' means the same as 'beautiful'. Both mean something looks nice.",
                    "type": "word_finding",
                    "options": ["pretty", "ugly", "old"]
                },
                {
                    "question": "Spell the word: M-O-T-H-E-R",
                    "answer": "mother",
                    "hint": "It's a word for 'mom'.",
                    "explanation": "M-O-T-H-E-R spells 'mother'. It's another word for 'mom'.",
                    "type": "spelling",
                    "word": "mother"
                },
            ],
            5: [  # Hard - 7+ letter words and advanced word finding
                {
                    "question": "Spell the word: T-E-A-C-H-E-R",
                    "answer": "teacher",
                    "hint": "It's someone who helps you learn at school.",
                    "explanation": "T-E-A-C-H-E-R spells 'teacher'. It's a person who helps students learn.",
                    "type": "spelling",
                    "word": "teacher"
                },
                {
                    "question": "Find the word that means 'brave': scared, courageous, worried",
                    "answer": "courageous",
                    "hint": "Which word means the same as 'brave'?",
                    "explanation": "'Courageous' means the same as 'brave'. Both mean not being afraid.",
                    "type": "word_finding",
                    "options": ["scared", "courageous", "worried"]
                },
                {
                    "question": "Spell the word: S-T-U-D-E-N-T",
                    "answer": "student",
                    "hint": "It's someone who goes to school to learn.",
                    "explanation": "S-T-U-D-E-N-T spells 'student'. It's a person who learns at school.",
                    "type": "spelling",
                    "word": "student"
                },
                {
                    "question": "Find the word that means 'tired': energetic, exhausted, active",
                    "answer": "exhausted",
                    "hint": "Which word means the same as 'tired'?",
                    "explanation": "'Exhausted' means the same as 'tired'. Both mean feeling very sleepy or worn out.",
                    "type": "word_finding",
                    "options": ["energetic", "exhausted", "active"]
                },
                {
                    "question": "Spell the word: B-E-A-U-T-I-F-U-L",
                    "answer": "beautiful",
                    "hint": "It means something looks very nice.",
                    "explanation": "B-E-A-U-T-I-F-U-L spells 'beautiful'. It means something looks very nice or pretty.",
                    "type": "spelling",
                    "word": "beautiful"
                },
            ]
        }
        return lessons
    
    def get_lesson(self, difficulty: int, lesson_index: Optional[int] = None) -> Optional[Dict]:
        """
        Get a lesson for the specified difficulty level.
        
        Args:
            difficulty: Difficulty level (1-5)
            lesson_index: Specific lesson index, or None for random selection
            
        Returns:
            Lesson dictionary or None if difficulty level doesn't exist
        """
        if difficulty not in self.lessons:
            # Clamp difficulty to valid range
            difficulty = max(1, min(5, difficulty))
        
        available_lessons = self.lessons[difficulty]
        if not available_lessons:
            return None
        
        if lesson_index is None:
            import random
            lesson_index = random.randint(0, len(available_lessons) - 1)
        else:
            lesson_index = lesson_index % len(available_lessons)
        
        return available_lessons[lesson_index].copy()
    
    def get_all_lessons_for_level(self, difficulty: int) -> List[Dict]:
        """
        Get all lessons for a difficulty level.
        
        Args:
            difficulty: Difficulty level (1-5)
            
        Returns:
            List of lesson dictionaries
        """
        if difficulty not in self.lessons:
            difficulty = max(1, min(5, difficulty))
        return self.lessons[difficulty].copy()
    
    def verify_answer(self, lesson: Dict, user_answer: str) -> bool:
        """
        Verify if user's answer is correct.
        
        Args:
            lesson: Lesson dictionary
            user_answer: User's answer as string
            
        Returns:
            True if correct, False otherwise
        """
        user_answer = user_answer.strip().lower()
        correct_answer = lesson["answer"].lower()
        return user_answer == correct_answer

