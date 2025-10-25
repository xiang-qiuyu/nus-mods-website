"""
Flask Backend API for NUSMods Timetable Optimizer
Run with: python backend.py
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from ortools.sat.python import cp_model
from typing import List, Dict, Tuple
from collections import defaultdict
import traceback
import openai
import os
from dotenv import load_dotenv
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend


# ==================== NUSMods API Handler ====================

class NUSModsAPI:
    """Fetches module data from NUSMods API"""
    
    BASE_URL = "https://api.nusmods.com/v2"
    ACADEMIC_YEAR = "2025-2026"
    
    @staticmethod
    def get_module_data(module_code: str) -> Dict:
        """Fetch module information including all lesson slots"""
        url = f"{NUSModsAPI.BASE_URL}/{NUSModsAPI.ACADEMIC_YEAR}/modules/{module_code}.json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {module_code}: {e}")
            return None
    
    @staticmethod
    def parse_lessons(module_data: Dict) -> Dict:
        """Parse lessons and group by lesson type and class number"""
        lessons_grouped = defaultdict(lambda: defaultdict(list))
        
        if not module_data or 'semesterData' not in module_data:
            return {}
        
        # Get first semester data (Semester 1)
        semester_data = None
        for sem in module_data['semesterData']:
            if sem['semester'] == 1:
                semester_data = sem
                break
        
        if not semester_data or 'timetable' not in semester_data:
            return {}
        
        # Group by lesson type, then by class number
        for lesson in semester_data['timetable']:
            lesson_type = lesson['lessonType']
            class_no = lesson['classNo']
            
            lessons_grouped[lesson_type][class_no].append({
                'day': lesson['day'],
                'startTime': lesson['startTime'],
                'endTime': lesson['endTime'],
                'venue': lesson['venue'],
                'weeks': lesson.get('weeks')
            })
        
        return {k: dict(v) for k, v in lessons_grouped.items()}


# ==================== Timetable Optimizer ====================

class TimetableOptimizer:
    """Optimization engine using Google OR-Tools CP-SAT solver"""
    
    DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    def __init__(self, modules: List[str], preferences: Dict):
        self.modules = modules
        self.preferences = preferences
        self.module_data = {}
        self.lessons_by_module = {}
        
    def fetch_module_data(self):
        """Fetch all module data from API"""
        print(f"Fetching module data for: {', '.join(self.modules)}")
        for module_code in self.modules:
            data = NUSModsAPI.get_module_data(module_code)
            if data:
                self.module_data[module_code] = data
                self.lessons_by_module[module_code] = NUSModsAPI.parse_lessons(data)
                print(f"✓ Loaded {module_code}: {data.get('title', 'Unknown')}")
            else:
                print(f"✗ Failed to load {module_code}")
    
    def _get_exam_interval_for_module(self, module_code: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Parse exam start datetime and compute exam end datetime for a module (if available).
        Returns (start_dt, end_dt) or None if no valid exam info.
        """
        data = self.module_data.get(module_code)
        if not data:
            data = NUSModsAPI.get_module_data(module_code)
            if data:
                self.module_data[module_code] = data
            else:
                print(f"  No data found for {module_code}")
                return None

        sem_data = data.get('semesterData', [])
        print(f"  {module_code}: Found {len(sem_data)} semester(s)")
        
        # Check ALL semesters, not just semester 1
        for s in sem_data:
            exam_date = s.get('examDate')
            exam_duration = s.get('examDuration')
            semester = s.get('semester')
            print(f"    Sem {semester}: examDate={exam_date}, duration={exam_duration}")
            
            if exam_date and exam_duration:
                try:
                    start_dt = datetime.fromisoformat(exam_date)
                except Exception:
                    try:
                        start_dt = datetime.fromisoformat(exam_date.replace('Z', '+00:00'))
                    except Exception:
                        print(f"    Failed to parse exam date: {exam_date}")
                        continue
                try:
                    duration_min = int(exam_duration)
                except Exception:
                    duration_min = 120
                end_dt = start_dt + timedelta(minutes=duration_min)
                print(f"    ✓ {module_code} exam: {start_dt} to {end_dt}")
                return (start_dt, end_dt)
        
        print(f"    No valid exam info for {module_code}")
        return None

    def find_exam_clashes(self, module_codes: List[str]) -> List[Dict]:
        """
        Returns list of exam clash groups:
        { 'modules': [..], 'overlap_start': ISO, 'overlap_end': ISO }
        """
        print(f"Checking exam clashes for: {module_codes}")
        intervals = []
        for mod in module_codes:
            interval = self._get_exam_interval_for_module(mod)
            if interval is not None:
                intervals.append({'module': mod, 'start': interval[0], 'end': interval[1]})

        print(f"Found {len(intervals)} modules with exam times")
        if len(intervals) < 2:
            print("Not enough modules with exam times to check for clashes")
            return []

        intervals.sort(key=lambda x: x['start'])

        groups: List[List[Dict]] = []
        current_group = [intervals[0]]
        current_end = intervals[0]['end']

        for it in intervals[1:]:
            if it['start'] < current_end:
                current_group.append(it)
                if it['end'] > current_end:
                    current_end = it['end']
            else:
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = [it]
                current_end = it['end']

        if len(current_group) > 1:
            groups.append(current_group)

        result = []
        for g in groups:
            overlap_start = min(item['start'] for item in g)
            overlap_end = max(item['end'] for item in g)
            result.append({
                'modules': [item['module'] for item in g],
                'overlap_start': overlap_start.isoformat(),
                'overlap_end': overlap_end.isoformat(),
            })
        return result
    
    @staticmethod
    def time_to_slot(time_str: str) -> int:
        """Convert time string (e.g., '0900') to slot index (30-min intervals from 8am)"""
        hours = int(time_str[:2])
        minutes = int(time_str[2:])
        return (hours - 8) * 2 + (minutes // 30)
    
    def optimize(self) -> List[Dict]:
        """Run optimization and return multiple timetable solutions"""
        self.fetch_module_data()
        
        if not self.lessons_by_module:
            return []
        
        # Check if all modules have lessons
        for module, lessons in self.lessons_by_module.items():
            if not lessons:
                print(f"Warning: {module} has no lessons available")
        # Validate blocked times if provided
        blocked_times = self.preferences.get('blockedTimes', [])
        if blocked_times and not self._validate_blocked_times(blocked_times):
            print("Warning: Invalid blocked times format, ignoring...")
            self.preferences['blockedTimes'] = []
        
        model = cp_model.CpModel()
        
        # Decision variables: for each module, lesson type, and class number
        # One variable per class number - if selected, ALL sessions of that class are included
        class_vars = {}
        all_selected_lessons = []  # Track all lesson sessions that will be in the timetable
        
        for module, lessons_by_type in self.lessons_by_module.items():
            if not lessons_by_type:
                continue
                
            class_vars[module] = {}
            for lesson_type, classes_dict in lessons_by_type.items():
                class_vars[module][lesson_type] = {}
                
                for class_no, sessions in classes_dict.items():
                    # Create ONE variable for this class (represents all its sessions)
                    var = model.NewBoolVar(f"{module}_{lesson_type}_{class_no}")
                    class_vars[module][lesson_type][class_no] = (var, sessions)
                    
                    # When this variable is true, ALL sessions are included
                    for session in sessions:
                        all_selected_lessons.append((var, session, module, lesson_type, class_no))
                
                # Constraint: exactly one class must be chosen per lesson type
                model.Add(sum(var for var, _ in class_vars[module][lesson_type].values()) == 1)
        
        if not class_vars:
            print("No valid class variables created")
            return []
        
        # Constraint: No overlapping classes
        self._add_no_overlap_constraints(model, all_selected_lessons)
        
        # Constraint: Block user-specified time slots (HARD CONSTRAINT)
        self._add_blocked_time_constraints(model, all_selected_lessons)

        # Soft constraints (preferences)
        objective_terms = []
        
        if self.preferences.get('noMorningClasses'):
            morning_penalty = self._add_no_morning_classes(model, all_selected_lessons)
            if morning_penalty is not None:
                objective_terms.append(morning_penalty)
        
        if self.preferences.get('freeFridays'):
            friday_penalty = self._add_free_fridays(model, all_selected_lessons)
            if friday_penalty is not None:
                objective_terms.append(friday_penalty)
        
        if self.preferences.get('lunchBreak'):
            lunch_penalty = self._add_lunch_break_constraint(model, all_selected_lessons)
            if lunch_penalty is not None:
                objective_terms.append(lunch_penalty)
        
        if objective_terms:
            model.Minimize(sum(objective_terms))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        solver.parameters.enumerate_all_solutions = True  # Key: enumerate all solutions
        
        class SolutionCollector(cp_model.CpSolverSolutionCallback):
            def __init__(self, variables):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self._variables = variables
                self.solutions = []
            
            def on_solution_callback(self):
                solution = {}
                for module, types in self._variables.items():
                    solution[module] = {}
                    for lesson_type, classes in types.items():
                        for class_no, (var, sessions) in classes.items():
                            if self.Value(var):
                                solution[module][lesson_type] = (class_no, sessions)
                                break
                self.solutions.append(solution)
        
        solution_collector = SolutionCollector(class_vars)
        status = solver.Solve(model, solution_collector)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print(f"Found {len(solution_collector.solutions)} solution(s)")
            all_timetables = self._format_solutions(solution_collector.solutions)
            all_timetables.sort(key=lambda x: x['score'], reverse=True)
            return all_timetables[:5]  # Return best 5
        else:
            print(f"No feasible solution found. Status: {status}")
            return []
    
    def _add_no_overlap_constraints(self, model, all_selected_lessons):
        """Ensure no two lesson sessions overlap"""
        for i in range(len(all_selected_lessons)):
            for j in range(i + 1, len(all_selected_lessons)):
                var1, session1, _, _, _ = all_selected_lessons[i]
                var2, session2, _, _, _ = all_selected_lessons[j]
                
                if self._lessons_overlap(session1, session2):
                    # If both classes are selected, sessions overlap - not allowed
                    model.Add(var1 + var2 <= 1)
    
    def _validate_blocked_times(self, blocked_times: List[Dict]) -> bool:
        """Validate blocked times format"""
        valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
        for blocked in blocked_times:
            # Check required fields
            if 'day' not in blocked or 'startTime' not in blocked or 'endTime' not in blocked:
                print(f"Invalid blocked time (missing fields): {blocked}")
                return False
        
            # Check valid day
            if blocked['day'] not in valid_days:
                print(f"Invalid day: {blocked['day']}")
                return False
        
            # Check time format (should be 4 digits)
            try:
                start = int(blocked['startTime'])
                end = int(blocked['endTime'])
                if not (0 <= start <= 2359 and 0 <= end <= 2359):
                    raise ValueError
                if start >= end:
                    print(f"Start time must be before end time: {blocked}")
                    return False
            except (ValueError, TypeError):
                print(f"Invalid time format: {blocked}")
                return False
    
        return True                
    
    def _add_blocked_time_constraints(self, model, all_selected_lessons):
        """
        Hard constraint: Prevent classes from being scheduled during blocked time slots.

        Expected format in preferences:
        'blockedTimes': [
            {'day': 'Monday', 'startTime': '1400', 'endTime': '1600'},
            {'day': 'Wednesday', 'startTime': '0900', 'endTime': '1000'},
            ...
        ]
        """
        blocked_times = self.preferences.get('blockedTimes', [])

        if not blocked_times:
            return  # No blocked times specified

        print(f"Processing {len(blocked_times)} blocked time slot(s)...")

        blocked_count = 0
        # all_selected_lessons is a list of (var, session, module, lesson_type, class_no)
        for var, session, module, lesson_type, class_no in all_selected_lessons:
            # Check if this session conflicts with any blocked time
            for blocked in blocked_times:
                if self._lesson_conflicts_with_blocked_time(session, blocked):
                    # This class cannot be selected (set to 0)
                    model.Add(var == 0)
                    blocked_count += 1
                    print(f"  ✗ Blocked: {module} {lesson_type} [{class_no}] on {session['day']} "
                        f"{session['startTime']}-{session['endTime']} (conflicts with blocked time)")
                    break  # No need to check other blocked times for this session

        print(f"Total sessions blocked: {blocked_count}")

    def _lesson_conflicts_with_blocked_time(self, lesson: Dict, blocked: Dict) -> bool:
        """Check if a lesson conflicts with a blocked time slot"""
        # Must be on the same day
        if lesson['day'] != blocked['day']:
            return False
    
        # Convert times to slots
        lesson_start = self.time_to_slot(lesson['startTime'])
        lesson_end = self.time_to_slot(lesson['endTime'])
        blocked_start = self.time_to_slot(blocked['startTime'])
        blocked_end = self.time_to_slot(blocked['endTime'])
    
        # Check if they overlap
        # Two intervals overlap if: NOT (one ends before other starts)
        return not (lesson_end <= blocked_start or blocked_end <= lesson_start)
    

    def _lessons_overlap(self, lesson1: Dict, lesson2: Dict) -> bool:
        """Check if two lessons overlap in time and weeks (if weeks info exists)."""

        # Different day => no overlap
        if lesson1.get('day') != lesson2.get('day'):
            return False

        start1 = self.time_to_slot(lesson1['startTime'])
        end1 = self.time_to_slot(lesson1['endTime'])
        start2 = self.time_to_slot(lesson2['startTime'])
        end2 = self.time_to_slot(lesson2['endTime'])

        time_overlap = not (end1 <= start2 or end2 <= start1)
        if not time_overlap:
            return False

        # If both lessons include explicit weeks info, check week intersection.
        weeks1 = lesson1.get('weeks')  # expected to be a list like [1,2,3] if parse provides it
        weeks2 = lesson2.get('weeks')

        if weeks1 is None or weeks2 is None:
            # Fallback behavior: no weeks info — treat as overlapping based on time only.
            # If you prefer conservative behavior (treat unknown weeks as conflict), change to `return True` here.
            return True

        try:
            set1 = set(int(w) for w in weeks1)
            set2 = set(int(w) for w in weeks2)
        except Exception:
            # If parsing fails, be conservative and treat as overlapping
            return True

        # Return true only if weeks intersect
        return len(set1.intersection(set2)) > 0
    
    def _add_no_morning_classes(self, model, all_selected_lessons):
        """Penalize classes before 10am"""
        morning_vars = []
        for var, session, _, _, _ in all_selected_lessons:
            start_slot = self.time_to_slot(session['startTime'])
            if start_slot < 4:  # Before 10am
                morning_vars.append(var)
        
        if morning_vars:
            penalty = model.NewIntVar(0, len(morning_vars) * 10, 'morning_penalty')
            model.Add(penalty == sum(morning_vars) * 10)
            return penalty
        return None
    
    def _add_free_fridays(self, model, all_selected_lessons):
        """Penalize Friday classes"""
        friday_vars = []
        for var, session, _, _, _ in all_selected_lessons:
            if session['day'] == 'Friday':
                friday_vars.append(var)
        
        if friday_vars:
            penalty = model.NewIntVar(0, len(friday_vars) * 15, 'friday_penalty')
            model.Add(penalty == sum(friday_vars) * 15)
            return penalty
        return None
    
    def _add_lunch_break_constraint(self, model, all_selected_lessons):
        """Penalize classes during lunch (12pm-2pm)"""
        lunch_vars = []
        for var, session, _, _, _ in all_selected_lessons:
            start_slot = self.time_to_slot(session['startTime'])
            end_slot = self.time_to_slot(session['endTime'])
            
            # Lunch time: 12pm = slot 8, 2pm = slot 12
            if not (end_slot <= 8 or start_slot >= 12):
                lunch_vars.append(var)
        
        if lunch_vars:
            penalty = model.NewIntVar(0, len(lunch_vars) * 8, 'lunch_penalty')
            model.Add(penalty == sum(lunch_vars) * 8)
            return penalty
        return None
    
    def _format_solutions(self, raw_solutions: List[Dict]) -> List[Dict]:
        """Format solutions into readable timetables"""
        formatted = []
        for idx, solution in enumerate(raw_solutions):
            schedule = []
            for module, lessons in solution.items():
                for lesson_type, (class_no, sessions) in lessons.items():
                    # Add ALL sessions for this class
                    for session in sessions:
                        schedule.append({
                            'module': module,
                            'type': lesson_type,
                            'classNo': class_no,
                            'day': session['day'],
                            'startTime': session['startTime'],
                            'endTime': session['endTime'],
                            'venue': session['venue'],
                            'weeks': session.get('weeks')
                        })
            
            # Sort by day and time
            schedule.sort(key=lambda x: (self.DAYS.index(x['day']), x['startTime']))
            
            formatted.append({
                'id': idx + 1,
                'schedule': schedule,
                'score': self._calculate_score(schedule),
                'tradeoffs': self._generate_tradeoffs(schedule)
            })
        
        return formatted
    
    def _calculate_score(self, schedule: List[Dict]) -> int:
        """Calculate timetable quality score"""
        score = 100
        
        if self.preferences.get('noMorningClasses'):
            morning_count = sum(1 for s in schedule if self.time_to_slot(s['startTime']) < 4)
            score -= morning_count * 5
        
        if self.preferences.get('freeFridays'):
            friday_count = sum(1 for s in schedule if s['day'] == 'Friday')
            score -= friday_count * 10
        
        if self.preferences.get('lunchBreak'):
            lunch_conflicts = self._get_lunch_conflicts(schedule)
            score -= len(lunch_conflicts) * 8
        
        return max(0, score)
    
    def _generate_tradeoffs(self, schedule: List[Dict]) -> List[str]:
        """Generate human-readable tradeoff explanations"""
        tradeoffs = []
        
        morning_classes = [s for s in schedule if self.time_to_slot(s['startTime']) < 4]
        if self.preferences.get('noMorningClasses'):
            if not morning_classes:
                tradeoffs.append('✓ No classes before 10am')
            else:
                tradeoffs.append(f'⚠ {len(morning_classes)} morning class(es) scheduled')
        
        friday_classes = [s for s in schedule if s['day'] == 'Friday']
        if self.preferences.get('freeFridays'):
            if not friday_classes:
                tradeoffs.append('✓ Fridays are completely free')
            else:
                tradeoffs.append(f'⚠ {len(friday_classes)} class(es) on Friday')
        
        if self.preferences.get('lunchBreak'):
            lunch_conflicts = self._get_lunch_conflicts(schedule)
            if not lunch_conflicts:
                tradeoffs.append('✓ Lunch breaks (12-2pm) preserved on all days')
            else:
                tradeoffs.append(f'⚠ Lunch conflicts on: {", ".join(lunch_conflicts)}')
        
        days_with_classes = set(s['day'] for s in schedule)
        tradeoffs.append(f'📅 Classes spread across {len(days_with_classes)} day(s)')
        
        return tradeoffs
    
    def _get_lunch_conflicts(self, schedule: List[Dict]) -> List[str]:
        """Get days with lunch conflicts"""
        days_schedule = defaultdict(list)
        for slot in schedule:
            days_schedule[slot['day']].append(slot)
        
        conflict_days = []
        for day, slots in days_schedule.items():
            for slot in slots:
                start = self.time_to_slot(slot['startTime'])
                end = self.time_to_slot(slot['endTime'])
                if not (end <= 8 or start >= 12):
                    conflict_days.append(day)
                    break
        
        return conflict_days

# ==================== ChatGPT Integration ====================

load_dotenv()  # Load environment variables from .env file

def generate_chatgpt_prompt(modules: List[str], preferences: Dict, blocked_times: List[Dict], user_notes: str = "") -> str:
    """
    Generate a comprehensive prompt for ChatGPT to create timetable schedules.
    """
    
    # Fetch module data
    module_details = {}
    for module_code in modules:
        data = NUSModsAPI.get_module_data(module_code)
        if data:
            lessons = NUSModsAPI.parse_lessons(data)
            module_details[module_code] = {
                'title': data.get('title', 'Unknown'),
                'lessons': lessons
            }
    
    # Build the prompt
    prompt = f"""You are an expert university academic advisor specializing in timetable optimization for NUS (National University of Singapore), 
    make sure that all classes,labs,seminar, or any other types are accounted for in the final timetable (you cant take a module without taking all its component classes!!!).
    

**TASK**: Create 3-5 optimal class schedules based on the following information.

**STUDENT'S MODULES**:
{json.dumps(modules, indent=2)}

**MODULE DETAILS**:
{json.dumps(module_details, indent=2)}

**STUDENT'S PREFERENCES**:
"""
    
    if preferences.get('noMorningClasses'):
        prompt += "- ✓ No classes before 10:00 AM\n"
    if preferences.get('freeFridays'):
        prompt += "- ✓ Keep Fridays completely free\n"
    if preferences.get('lunchBreak'):
        prompt += "- ✓ Preserve lunch break (12:00-14:00) every day\n"
    if preferences.get('compactSchedule'):
        prompt += "- ✓ Minimize gaps between classes (compact schedule)\n"
    if preferences.get('minimizeTravel'):
        prompt += "- ✓ Minimize walking distance between venues\n"
    
    if blocked_times:
        prompt += f"\n**BLOCKED TIME SLOTS** (Student is unavailable):\n"
        for blocked in blocked_times:
            prompt += f"- {blocked['day']} {blocked['startTime']}-{blocked['endTime']}\n"
    
    if user_notes:
        prompt += f"\n**ADDITIONAL NOTES FROM STUDENT**:\n{user_notes}\n"
    
    prompt += """

**OUTPUT REQUIREMENTS**:

For each schedule option, provide:

1. **Schedule ID**: Option 1, Option 2, etc.
2. **Quality Score**: 0-100 (explain scoring)
3. **Weekly Timetable**: Present as a table with columns:
   - Day | Time | Module | Type (Lecture/Tutorial/Lab) | Class No | Venue

4. **Constraint Analysis**:
   - ✅ Satisfied preferences
   - ⚠️ Compromised preferences (explain why)
   - ❌ Violations (if any)

5. **Trade-offs**: Explain key decisions made

6. **Exam Schedule**: Check for exam clashes

7. **Recommendation**: Which option is best and why

**IMPORTANT RULES**:
- Each module requires selecting ONE class number per lesson type (Lecture, Tutorial, Lab, etc.)
- Classes from the same module/type CANNOT overlap
- Classes must not conflict with blocked time slots
- Explain ALL scheduling decisions
- If no feasible schedule exists, explain why and suggest alternatives

Generate 3-5 schedule options now, ranked by quality score.
"""
    
    return prompt


@app.route('/api/optimize-chatgpt', methods=['POST'])
def optimize_with_chatgpt():
    """
    ChatGPT-powered timetable optimization endpoint
    
    Expected JSON body:
    {
        "modules": ["CS1010", "MA1521"],
        "preferences": {...},
        "blockedTimes": [...],
        "userNotes": "I prefer morning classes and need time for gym in the evening"
    }
    """
    
    try:
        data = request.json
        modules = data.get('modules', [])
        preferences = data.get('preferences', {})
        blocked_times = data.get('blockedTimes', [])
        user_notes = data.get('userNotes', '')
        
        print(f"\n{'='*60}")
        print(f"ChatGPT Optimization Request:")
        print(f"Modules: {modules}")
        print(f"User Notes: {user_notes}")
        print(f"{'='*60}\n")
        
        if not modules:
            return jsonify({'error': 'No modules provided'}), 400
        
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return jsonify({
                'error': 'OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file'
            }), 500
        
        # Generate prompt
        prompt = generate_chatgpt_prompt(modules, preferences, blocked_times, user_notes)
        
        # Call OpenAI API
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo" for faster/cheaper
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert university academic advisor specializing in timetable optimization. Provide detailed, well-structured schedule recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        chatgpt_response = response.choices[0].message.content
        
        print(f"\n{'='*60}")
        print(f"ChatGPT Response Received")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'response': chatgpt_response,
            'prompt_used': prompt,  # For debugging
            'model': response.model,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        })
        
    except openai.APIError as e:
        print(f"OpenAI API Error: {str(e)}")
        return jsonify({'error': f'OpenAI API Error: {str(e)}'}), 500
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Flask API Routes ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Backend is running'})


@app.route('/api/optimize', methods=['POST'])
def optimize_timetable():
    """Main optimization endpoint
    
    Expected JSON body:
    {
        "modules": ["CS1010", "MA1521"],
        "preferences": {
            "noMorningClasses": true,
            "freeFridays": true,
            "lunchBreak": true,
            "minimizeTravel": false,
            "compactSchedule": true,
            "blockedTimes": [
                {"day": "Monday", "startTime": "1400", "endTime": "1600"},
                {"day": "Wednesday", "startTime": "0900", "endTime": "1000"}
            ]
        }
    }
    """
    
    try:
        data = request.json
        modules = data.get('modules', [])
        preferences = data.get('preferences', {})

        print(f"\n{'='*60}")
        print(f"Received optimization request:")
        print(f"Modules: {modules}")
        print(f"Preferences: {preferences}")
        print(f"{'='*60}\n")

        if not modules:
            return jsonify({'error': 'No modules provided'}), 400
        optimizer = TimetableOptimizer(modules, preferences)
        # Ensure module_data and lessons_by_module are populated for exam checking
        optimizer.fetch_module_data()

        exam_clashes = optimizer.find_exam_clashes(modules)
        print(f"Exam clash detection result: {len(exam_clashes)} clashes found")
        
        # Always block if exam clashes are detected (minimal change: remove preference check)
        if exam_clashes:
            return jsonify({
                'error': 'Exam clashes detected between your selected modules',
                'examClashes': exam_clashes,
                'count': len(exam_clashes),
            }), 409

        timetables = optimizer.optimize()

        if not timetables:
            return jsonify({
                'error': 'No feasible timetables found. This might be due to:\n'
                         '- Module codes not available for current semester\n'
                         '- Too many conflicting constraints\n'
                         '- Invalid module codes',
                'examClashes': exam_clashes,
            }), 404

        return jsonify({
            'success': True,
            'timetables': timetables,
            'count': len(timetables),
            'examClashes': exam_clashes,
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/modules/<module_code>', methods=['GET'])
def get_module_info(module_code):
    try:
        data = NUSModsAPI.get_module_data(module_code)
        if not data:
            return jsonify({'error': f'Module {module_code} not found'}), 404
        
        lessons = NUSModsAPI.parse_lessons(data)
        
        return jsonify({
            'code': module_code,
            'title': data.get('title'),
            'lessons': lessons
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NUSMods Timetable Optimizer Backend")
    print("="*60)
    print("Server starting on http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)