"""
Math lesson content for the AI Tutor.
Provides single-digit addition and subtraction problems with hints and step-by-step breakdowns.
"""
from typing import Dict, List, Tuple, Optional


class MathLessons:
    """Math lesson content organized by difficulty level."""
    
    def __init__(self):
        """Initialize math lessons database."""
        self.lessons = self._initialize_lessons()
    
    def _initialize_lessons(self) -> Dict[int, List[Dict]]:
        """
        Initialize lesson database with questions, answers, hints, and steps.
        
        Returns:
            Dictionary mapping difficulty levels to lists of lesson items
        """
        lessons = {
            1: [  # Very Easy - Single digit addition (sums < 10)
                {
                    "question": "What is 2 + 3?",
                    "answer": 5,
                    "hint": "Count: 2, then count 3 more. How many total?",
                    "steps": ["Start with 2", "Add 3 more", "Count all together: 2 + 3 = 5"],
                    "type": "addition"
                },
                {
                    "question": "What is 1 + 4?",
                    "answer": 5,
                    "hint": "Start with 1, add 4 more.",
                    "steps": ["Start with 1", "Add 4 more", "1 + 4 = 5"],
                    "type": "addition"
                },
                {
                    "question": "What is 3 + 2?",
                    "answer": 5,
                    "hint": "3 plus 2 equals?",
                    "steps": ["Start with 3", "Add 2 more", "3 + 2 = 5"],
                    "type": "addition"
                },
                {
                    "question": "What is 4 + 1?",
                    "answer": 5,
                    "hint": "4 and 1 make how many?",
                    "steps": ["Start with 4", "Add 1 more", "4 + 1 = 5"],
                    "type": "addition"
                },
                {
                    "question": "What is 2 + 2?",
                    "answer": 4,
                    "hint": "Two plus two equals?",
                    "steps": ["Start with 2", "Add 2 more", "2 + 2 = 4"],
                    "type": "addition"
                },
            ],
            2: [  # Easy - Single digit addition (sums 5-10)
                {
                    "question": "What is 5 + 3?",
                    "answer": 8,
                    "hint": "Start with 5, count 3 more.",
                    "steps": ["Start with 5", "Add 3 more", "5 + 3 = 8"],
                    "type": "addition"
                },
                {
                    "question": "What is 4 + 4?",
                    "answer": 8,
                    "hint": "Four plus four equals?",
                    "steps": ["Start with 4", "Add 4 more", "4 + 4 = 8"],
                    "type": "addition"
                },
                {
                    "question": "What is 6 + 2?",
                    "answer": 8,
                    "hint": "6 and 2 make how many?",
                    "steps": ["Start with 6", "Add 2 more", "6 + 2 = 8"],
                    "type": "addition"
                },
                {
                    "question": "What is 3 + 5?",
                    "answer": 8,
                    "hint": "3 plus 5 equals?",
                    "steps": ["Start with 3", "Add 5 more", "3 + 5 = 8"],
                    "type": "addition"
                },
                {
                    "question": "What is 7 + 1?",
                    "answer": 8,
                    "hint": "7 and 1 make?",
                    "steps": ["Start with 7", "Add 1 more", "7 + 1 = 8"],
                    "type": "addition"
                },
            ],
            3: [  # Medium - Single digit addition (sums up to 15) and simple subtraction
                {
                    "question": "What is 7 + 5?",
                    "answer": 12,
                    "hint": "Start with 7, add 5. You can break 5 into 3+2.",
                    "steps": ["Start with 7", "Add 3 first: 7 + 3 = 10", "Then add 2 more: 10 + 2 = 12", "So 7 + 5 = 12"],
                    "type": "addition"
                },
                {
                    "question": "What is 9 + 4?",
                    "answer": 13,
                    "hint": "9 is close to 10. Add 4 to 9.",
                    "steps": ["Start with 9", "9 + 1 = 10", "Then add 3 more: 10 + 3 = 13", "So 9 + 4 = 13"],
                    "type": "addition"
                },
                {
                    "question": "What is 8 - 3?",
                    "answer": 5,
                    "hint": "Start with 8, take away 3.",
                    "steps": ["Start with 8", "Take away 3", "Count what's left: 8 - 3 = 5"],
                    "type": "subtraction"
                },
                {
                    "question": "What is 10 - 4?",
                    "answer": 6,
                    "hint": "10 minus 4 equals?",
                    "steps": ["Start with 10", "Take away 4", "10 - 4 = 6"],
                    "type": "subtraction"
                },
                {
                    "question": "What is 6 + 6?",
                    "answer": 12,
                    "hint": "Double 6 equals?",
                    "steps": ["6 + 6 means two groups of 6", "6 + 6 = 12"],
                    "type": "addition"
                },
            ],
            4: [  # Medium-Hard - Two-digit addition and subtraction
                {
                    "question": "What is 12 + 5?",
                    "answer": 17,
                    "hint": "Add 5 to 12. Start with the ones place.",
                    "steps": ["12 has 1 ten and 2 ones", "Add 5 ones: 2 + 5 = 7", "Keep the 1 ten", "So 12 + 5 = 17"],
                    "type": "addition"
                },
                {
                    "question": "What is 15 - 7?",
                    "answer": 8,
                    "hint": "15 minus 7. You can think: what plus 7 equals 15?",
                    "steps": ["Start with 15", "Take away 7", "15 - 7 = 8", "Check: 8 + 7 = 15"],
                    "type": "subtraction"
                },
                {
                    "question": "What is 9 + 8?",
                    "answer": 17,
                    "hint": "9 is close to 10. Add 8 to 9.",
                    "steps": ["9 + 1 = 10", "Then add 7 more: 10 + 7 = 17", "So 9 + 8 = 17"],
                    "type": "addition"
                },
                {
                    "question": "What is 14 - 6?",
                    "answer": 8,
                    "hint": "14 minus 6 equals?",
                    "steps": ["Start with 14", "Take away 6", "14 - 6 = 8"],
                    "type": "subtraction"
                },
                {
                    "question": "What is 11 + 6?",
                    "answer": 17,
                    "hint": "11 plus 6 equals?",
                    "steps": ["11 + 6", "1 + 6 = 7 in ones place", "Keep the 1 ten", "11 + 6 = 17"],
                    "type": "addition"
                },
            ],
            5: [  # Hard - More complex two-digit operations
                {
                    "question": "What is 18 - 9?",
                    "answer": 9,
                    "hint": "18 minus 9. Think: what number plus 9 equals 18?",
                    "steps": ["18 - 9", "Think: 9 + ? = 18", "9 + 9 = 18", "So 18 - 9 = 9"],
                    "type": "subtraction"
                },
                {
                    "question": "What is 13 + 7?",
                    "answer": 20,
                    "hint": "13 plus 7. Add the ones first.",
                    "steps": ["13 + 7", "3 + 7 = 10", "10 + 10 = 20", "So 13 + 7 = 20"],
                    "type": "addition"
                },
                {
                    "question": "What is 16 - 8?",
                    "answer": 8,
                    "hint": "16 minus 8 equals?",
                    "steps": ["16 - 8", "Think: 8 + ? = 16", "8 + 8 = 16", "So 16 - 8 = 8"],
                    "type": "subtraction"
                },
                {
                    "question": "What is 15 + 5?",
                    "answer": 20,
                    "hint": "15 plus 5 makes a round number.",
                    "steps": ["15 + 5", "5 + 5 = 10", "10 + 10 = 20", "So 15 + 5 = 20"],
                    "type": "addition"
                },
                {
                    "question": "What is 19 - 11?",
                    "answer": 8,
                    "hint": "19 minus 11. Subtract tens first, then ones.",
                    "steps": ["19 - 11", "19 has 1 ten and 9 ones", "11 has 1 ten and 1 one", "1 ten - 1 ten = 0", "9 ones - 1 one = 8", "So 19 - 11 = 8"],
                    "type": "subtraction"
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
        try:
            user_num = int(user_answer.strip())
            return user_num == lesson["answer"]
        except ValueError:
            return False

