"""
Content engine that adapts teaching strategy based on detected emotions.
"""
from typing import Dict, Optional, Tuple
from modules.math_lessons import MathLessons
from modules.english_lessons import EnglishLessons


class ContentEngine:
    """Adapts educational content based on student emotions."""
    
    # Emotion to strategy mapping
    EMOTION_STRATEGIES = {
        "happy": {
            "action": "increase_difficulty",
            "message": "Great! You're doing well! Let's try something a bit more challenging.",
            "difficulty_change": 0.3,  # Increase difficulty slightly
            "hint_level": "minimal"  # Less hints needed
        },
        "neutral": {
            "action": "continue",
            "message": "Let's continue with the current level. You're doing fine!",
            "difficulty_change": 0.0,
            "hint_level": "normal"
        },
        "sad": {
            "action": "slow_pace",
            "message": "It's okay, take your time. Here's a small hint to help you.",
            "difficulty_change": -0.2,  # Slightly easier
            "hint_level": "small_hint"
        },
        "frustrated": {
            "action": "break_into_steps",
            "message": "Let's break this down into smaller steps. You can do it!",
            "difficulty_change": -0.3,  # Make it easier
            "hint_level": "step_by_step"
        },
        "angry": {
            "action": "calm_and_easier",
            "message": "Take a deep breath. Let's try an easier task together.",
            "difficulty_change": -0.5,  # Much easier
            "hint_level": "full_help"
        },
        "confused": {
            "action": "explain_and_example",
            "message": "Let me explain this with an example. Here's how it works:",
            "difficulty_change": -0.2,  # Slightly easier
            "hint_level": "explanation"
        }
    }
    
    def __init__(self):
        """Initialize content engine with lesson modules."""
        self.math_lessons = MathLessons()
        self.english_lessons = EnglishLessons()
        self.current_difficulty = 2  # Start at moderate level (1-5 scale)
        self.current_subject = "math"  # Default subject
        self.current_lesson = None
        self.lesson_history = []  # Track lesson progression
    
    def get_strategy(self, emotion: str) -> Dict:
        """
        Get teaching strategy for detected emotion.
        
        Args:
            emotion: Detected emotion label
            
        Returns:
            Strategy dictionary with action, message, and parameters
        """
        # Default to neutral if emotion not recognized
        emotion = emotion.lower()
        if emotion not in self.EMOTION_STRATEGIES:
            emotion = "neutral"
        
        return self.EMOTION_STRATEGIES[emotion].copy()
    
    def update_difficulty(self, emotion: str):
        """
        Update difficulty level based on emotion.
        
        Args:
            emotion: Detected emotion label
        """
        strategy = self.get_strategy(emotion)
        change = strategy["difficulty_change"]
        
        # Update difficulty (clamped to 1-5 range)
        new_difficulty = self.current_difficulty + change
        self.current_difficulty = max(1.0, min(5.0, new_difficulty))
        
        # Round to nearest integer for lesson selection
        self.current_difficulty = round(self.current_difficulty)
    
    def select_lesson(self, emotion: str, subject: Optional[str] = None) -> Dict:
        """
        Select next lesson based on emotion and current state.
        
        Args:
            emotion: Detected emotion label
            subject: Subject name ("math" or "english"), or None to use current
            
        Returns:
            Lesson dictionary with question, answer, hints, etc.
        """
        # Update subject if provided
        if subject is not None:
            self.current_subject = subject.lower()
        
        # Update difficulty based on emotion
        self.update_difficulty(emotion)
        
        # Get strategy
        strategy = self.get_strategy(emotion)
        
        # Select lesson from appropriate module
        if self.current_subject == "math":
            lesson = self.math_lessons.get_lesson(self.current_difficulty)
        elif self.current_subject == "english":
            lesson = self.english_lessons.get_lesson(self.current_difficulty)
        else:
            # Default to math
            lesson = self.math_lessons.get_lesson(self.current_difficulty)
        
        if lesson is None:
            # Fallback to difficulty 2 if no lesson found
            if self.current_subject == "math":
                lesson = self.math_lessons.get_lesson(2)
            else:
                lesson = self.english_lessons.get_lesson(2)
        
        # Add strategy information to lesson
        lesson["strategy"] = strategy
        lesson["difficulty"] = self.current_difficulty
        lesson["subject"] = self.current_subject
        
        # Store current lesson
        self.current_lesson = lesson
        
        # Add to history
        self.lesson_history.append({
            "emotion": emotion,
            "difficulty": self.current_difficulty,
            "subject": self.current_subject,
            "lesson": lesson.get("question", "")
        })
        
        return lesson
    
    def get_adaptive_content(self, emotion: str, subject: Optional[str] = None) -> Dict:
        """
        Get adaptive content with strategy message and lesson.
        
        Args:
            emotion: Detected emotion label
            subject: Subject name ("math" or "english"), or None to use current
            
        Returns:
            Dictionary with strategy message, lesson, and metadata
        """
        strategy = self.get_strategy(emotion)
        lesson = self.select_lesson(emotion, subject)
        
        # Format content based on hint level
        content = {
            "strategy_message": strategy["message"],
            "lesson": lesson,
            "difficulty": self.current_difficulty,
            "subject": self.current_subject,
            "emotion": emotion,
            "hint_level": strategy["hint_level"]
        }
        
        # Add hints based on hint level
        if strategy["hint_level"] == "minimal":
            content["hint"] = None  # No hint for happy students
        elif strategy["hint_level"] == "small_hint":
            content["hint"] = lesson.get("hint", "")
        elif strategy["hint_level"] == "step_by_step":
            content["hint"] = lesson.get("hint", "")
            content["steps"] = lesson.get("steps", [])
        elif strategy["hint_level"] == "full_help":
            content["hint"] = lesson.get("hint", "")
            content["steps"] = lesson.get("steps", [])
            if "explanation" in lesson:
                content["explanation"] = lesson["explanation"]
        elif strategy["hint_level"] == "explanation":
            content["hint"] = lesson.get("hint", "")
            if "explanation" in lesson:
                content["explanation"] = lesson["explanation"]
            if "steps" in lesson:
                content["steps"] = lesson["steps"]
        else:  # normal
            content["hint"] = None  # Available on request
        
        return content
    
    def verify_answer(self, user_answer: str) -> Tuple[bool, Optional[str]]:
        """
        Verify user's answer to current lesson.
        
        Args:
            user_answer: User's answer string
            
        Returns:
            Tuple of (is_correct, feedback_message)
        """
        if self.current_lesson is None:
            return False, "No lesson active. Please start a lesson first."
        
        if self.current_subject == "math":
            is_correct = self.math_lessons.verify_answer(self.current_lesson, user_answer)
        else:
            is_correct = self.english_lessons.verify_answer(self.current_lesson, user_answer)
        
        if is_correct:
            feedback = "Excellent! That's correct! 🎉"
            # Slightly increase difficulty for next question
            self.current_difficulty = min(5, self.current_difficulty + 0.2)
        else:
            feedback = f"Not quite. The correct answer is: {self.current_lesson['answer']}. Let's try another one!"
            # Slightly decrease difficulty for next question
            self.current_difficulty = max(1, self.current_difficulty - 0.2)
        
        return is_correct, feedback
    
    def reset(self):
        """Reset content engine to initial state."""
        self.current_difficulty = 2
        self.current_subject = "math"
        self.current_lesson = None
        self.lesson_history = []

