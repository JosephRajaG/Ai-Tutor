"""
Main Streamlit application for AI Tutor with Emotion-Adaptive Learning.
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os

from utils.camera import Camera
from utils.emotion_detector import EmotionDetector
from modules.content_engine import ContentEngine


# Page configuration
st.set_page_config(
    page_title="AI Tutor - Emotion Adaptive Learning",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "camera" not in st.session_state:
    st.session_state.camera = None
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "emotion_detector" not in st.session_state:
    st.session_state.emotion_detector = None
if "content_engine" not in st.session_state:
    st.session_state.content_engine = ContentEngine()
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"
if "emotion_confidence" not in st.session_state:
    st.session_state.emotion_confidence = 0.0
if "current_content" not in st.session_state:
    st.session_state.current_content = None
if "last_emotion_update" not in st.session_state:
    st.session_state.last_emotion_update = 0
if "last_frame_update" not in st.session_state:
    st.session_state.last_frame_update = 0
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True  # Enable by default for continuous monitoring
if "frame_placeholder" not in st.session_state:
    st.session_state.frame_placeholder = None
if "user_answer" not in st.session_state:
    st.session_state.user_answer = ""
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = ""


def initialize_emotion_detector():
    """Initialize emotion detector if not already done."""
    if st.session_state.emotion_detector is None:
        try:
            st.session_state.emotion_detector = EmotionDetector("models/emotion.onnx")
            return True
        except FileNotFoundError as e:
            st.error(f"❌ {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Error loading emotion detector: {str(e)}")
            return False
    return True


def start_camera():
    """Start camera capture."""
    if st.session_state.camera is None:
        st.session_state.camera = Camera()
    
    if st.session_state.camera.start():
        st.session_state.camera_active = True
        st.success("✅ Camera started!")
        return True
    else:
        st.error("❌ Failed to start camera. Please check if camera is available.")
        return False


def stop_camera():
    """Stop camera capture."""
    if st.session_state.camera is not None:
        st.session_state.camera.stop()
        st.session_state.camera_active = False
        st.info("📷 Camera stopped.")


def main():
    """Main application."""
    st.title("📚 AI Tutor - Emotion Adaptive Learning")
    st.markdown("---")
    
    # Check if model exists
    model_exists = os.path.exists("models/emotion.onnx")
    if not model_exists:
        st.warning("⚠️ **Model file not found!** Please place `emotion.onnx` in the `models/` directory.")
        st.stop()
    
    # Initialize emotion detector
    if not initialize_emotion_detector():
        st.stop()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎮 Controls")
        
        # Camera controls
        st.subheader("📷 Camera")
        if not st.session_state.camera_active:
            if st.button("▶️ Start Camera", type="primary", width='stretch'):
                start_camera()
        else:
            if st.button("⏹️ Stop Camera", width='stretch'):
                stop_camera()
            st.markdown("---")
            # Auto-refresh toggle (enabled by default for continuous monitoring)
            auto_refresh = st.checkbox("🔄 Continuous Monitoring", value=st.session_state.auto_refresh, key="auto_refresh_checkbox")
            st.session_state.auto_refresh = auto_refresh
            if auto_refresh:
                st.caption("📹 Camera is actively monitoring your reactions")
            else:
                st.caption("📸 Click 'Capture Frame' to update manually")
                if st.button("📸 Capture Frame", width='stretch'):
                    st.rerun()
        
        st.markdown("---")
        
        # Subject selection
        st.subheader("📖 Subject")
        subject = st.radio(
            "Choose a subject:",
            ["Math", "English"],
            index=0 if st.session_state.content_engine.current_subject == "math" else 1,
            key="subject_selector"
        )
        st.session_state.content_engine.current_subject = subject.lower()
        
        st.markdown("---")
        
        # Current status
        st.subheader("📊 Status")
        st.metric("Difficulty Level", f"{st.session_state.content_engine.current_difficulty}/5")
        
        # Display emotion with color coding
        emotion = st.session_state.current_emotion.title()
        confidence = st.session_state.emotion_confidence
        
        # Color code emotions
        emotion_colors = {
            "Happy": "🟢",
            "Sad": "🔵",
            "Frustrated": "🟠",
            "Angry": "🔴",
            "Confused": "🟡",
            "Neutral": "⚪"
        }
        emotion_icon = emotion_colors.get(emotion, "⚪")
        st.metric(f"{emotion_icon} Current Emotion", f"{emotion} ({confidence:.0%})")
        
        # Show confidence bar
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Reset Session", width='stretch'):
            st.session_state.content_engine.reset()
            st.session_state.current_content = None
            st.session_state.user_answer = ""
            st.session_state.show_feedback = False
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📹 Live Camera Feed")
        
        if st.session_state.camera_active:
            # Capture frame
            frame = st.session_state.camera.get_frame_rgb()
            
            if frame is not None:
                # Convert frame to PIL Image for more stable display
                from PIL import Image as PILImage
                pil_image = PILImage.fromarray(frame)
                
                # Display frame directly (more stable than placeholder)
                st.image(pil_image, channels="RGB", width='stretch')
                
                # Run emotion detection
                emotion, confidence, prob_dict = st.session_state.emotion_detector.predict_emotion(frame)
                
                # Only update if we got a valid emotion with reasonable confidence
                if emotion and confidence > 0.1:
                    # Check if emotion actually changed
                    emotion_changed = (st.session_state.current_emotion != emotion)
                    st.session_state.current_emotion = emotion
                    st.session_state.emotion_confidence = confidence
                    
                    # Update content more responsively - if emotion changed or enough time passed
                    current_time = time.time()
                    time_since_update = current_time - st.session_state.last_emotion_update
                    
                    if (emotion_changed or 
                        time_since_update > 2.0 or  # Update every 2 seconds even if same emotion
                        st.session_state.current_content is None):
                        st.session_state.current_content = st.session_state.content_engine.get_adaptive_content(
                            emotion, 
                            st.session_state.content_engine.current_subject
                        )
                        st.session_state.last_emotion_update = current_time
                
                # Display emotion probabilities for debugging (optional, can be removed)
                with st.expander("🔍 Emotion Probabilities (Debug)", expanded=False):
                    if prob_dict:
                        for emo, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
                            st.progress(prob, text=f"{emo.title()}: {prob:.1%}")
                
                # Auto-refresh if enabled for continuous monitoring (every 0.2 seconds)
                if st.session_state.auto_refresh:
                    current_time = time.time()
                    if current_time - st.session_state.last_frame_update > 0.2:
                        st.session_state.last_frame_update = current_time
                        time.sleep(0.05)  # Small delay
                        st.rerun()
                # If auto-refresh is disabled, the frame will only update when user clicks "Capture Frame"
            else:
                st.warning("⚠️ Could not capture frame from camera.")
        else:
            st.info("👆 Click 'Start Camera' in the sidebar to begin!")
            # Show static placeholder
            placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
            st.image(placeholder_img, channels="RGB", width='stretch')
    
    with col2:
        st.header("📚 Learning Content")
        
        if st.session_state.current_content is not None:
            content = st.session_state.current_content
            
            # Display strategy message
            st.info(f"💡 **Tutor:** {content['strategy_message']}")
            
            # Display current lesson
            if content['lesson']:
                lesson = content['lesson']
                
                st.markdown("### 📝 Current Question")
                st.markdown(f"**{lesson.get('question', 'No question available')}**")
                
                # Display explanation if available (for confused emotion)
                if "explanation" in content and content["explanation"]:
                    with st.expander("📖 Explanation"):
                        st.markdown(content["explanation"])
                
                # Display steps if available (for frustrated emotion)
                if "steps" in content and content["steps"]:
                    with st.expander("🔢 Step-by-Step Guide"):
                        for i, step in enumerate(content["steps"], 1):
                            st.markdown(f"{i}. {step}")
                
                # Answer input
                st.markdown("### ✍️ Your Answer")
                user_input = st.text_input(
                    "Enter your answer:",
                    value=st.session_state.user_answer,
                    key="answer_input"
                )
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ Submit Answer", type="primary", width='stretch'):
                        if user_input.strip():
                            is_correct, feedback = st.session_state.content_engine.verify_answer(user_input)
                            st.session_state.feedback_message = feedback
                            st.session_state.show_feedback = True
                            st.session_state.user_answer = ""  # Clear input
                            # Get new lesson
                            st.session_state.current_content = st.session_state.content_engine.get_adaptive_content(
                                st.session_state.current_emotion,
                                st.session_state.content_engine.current_subject
                            )
                            st.rerun()
                        else:
                            st.warning("Please enter an answer first!")
                
                with col_btn2:
                    if st.button("💡 Show Hint", width='stretch'):
                        if "hint" in content and content["hint"]:
                            st.info(f"💡 **Hint:** {content['hint']}")
                        elif lesson.get("hint"):
                            st.info(f"💡 **Hint:** {lesson['hint']}")
                        else:
                            st.info("No hint available for this question.")
                
                # Display feedback
                if st.session_state.show_feedback:
                    if "correct" in st.session_state.feedback_message.lower() or "excellent" in st.session_state.feedback_message.lower():
                        st.success(f"🎉 {st.session_state.feedback_message}")
                    else:
                        st.warning(f"📝 {st.session_state.feedback_message}")
                    
                    if st.button("Continue", width='stretch'):
                        st.session_state.show_feedback = False
                        st.rerun()
        else:
            st.info("👆 Start the camera to begin learning! The tutor will detect your emotions and adapt the content.")
            
            # Show sample content
            st.markdown("### 📋 How it works:")
            st.markdown("""
            1. **Start the camera** to enable emotion detection
            2. **Choose a subject** (Math or English)
            3. The AI Tutor will:
               - Detect your facial emotions in real-time
               - Adapt the difficulty and teaching style
               - Provide personalized hints and explanations
            4. **Answer questions** and get instant feedback
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>AI Tutor with Emotion-Adaptive Learning | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

