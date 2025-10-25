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
        self.venue_coordinates = self._load_venue_coordinates()
        
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
    
    def _load_venue_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """Load venue coordinates for distance calculation"""
        return {
            # Engineering Faculty (Kent Ridge Campus)
            'E1': (1.2966, 103.7764),
            'E2': (1.2966, 103.7764),
            'E3': (1.2966, 103.7764),
            'E4': (1.2966, 103.7764),
            'E5': (1.2966, 103.7764),
            'EA': (1.2966, 103.7764),
            'EB': (1.2966, 103.7764),
            'EC': (1.2966, 103.7764),
            'ED': (1.2966, 103.7764),
            'EE': (1.2966, 103.7764),
            'EF': (1.2966, 103.7764),
            'EG': (1.2966, 103.7764),
            'EH': (1.2966, 103.7764),
            'EI': (1.2966, 103.7764),
            'EJ': (1.2966, 103.7764),
            'EK': (1.2966, 103.7764),
            'EL': (1.2966, 103.7764),
            'EM': (1.2966, 103.7764),
            'EN': (1.2966, 103.7764),
            'EO': (1.2966, 103.7764),
            'EP': (1.2966, 103.7764),
            'EQ': (1.2966, 103.7764),
            'ER': (1.2966, 103.7764),
            'ES': (1.2966, 103.7764),
            'ET': (1.2966, 103.7764),
            'EU': (1.2966, 103.7764),
            'EV': (1.2966, 103.7764),
            'EW': (1.2966, 103.7764),
            'EX': (1.2966, 103.7764),
            'EY': (1.2966, 103.7764),
            'EZ': (1.2966, 103.7764),
            
            # Science Faculty (Kent Ridge Campus)
            'S1': (1.2966, 103.7764),
            'S2': (1.2966, 103.7764),
            'S3': (1.2966, 103.7764),
            'S4': (1.2966, 103.7764),
            'S5': (1.2966, 103.7764),
            'SA': (1.2966, 103.7764),
            'SB': (1.2966, 103.7764),
            'SC': (1.2966, 103.7764),
            'SD': (1.2966, 103.7764),
            'SE': (1.2966, 103.7764),
            'SF': (1.2966, 103.7764),
            'SG': (1.2966, 103.7764),
            'SH': (1.2966, 103.7764),
            'SI': (1.2966, 103.7764),
            'SJ': (1.2966, 103.7764),
            'SK': (1.2966, 103.7764),
            'SL': (1.2966, 103.7764),
            'SM': (1.2966, 103.7764),
            'SN': (1.2966, 103.7764),
            'SO': (1.2966, 103.7764),
            'SP': (1.2966, 103.7764),
            'SQ': (1.2966, 103.7764),
            'SR': (1.2966, 103.7764),
            'SS': (1.2966, 103.7764),
            'ST': (1.2966, 103.7764),
            'SU': (1.2966, 103.7764),
            'SV': (1.2966, 103.7764),
            'SW': (1.2966, 103.7764),
            'SX': (1.2966, 103.7764),
            'SY': (1.2966, 103.7764),
            'SZ': (1.2966, 103.7764),
            
            # Arts & Social Sciences (Kent Ridge Campus)
            'AS1': (1.2966, 103.7764),
            'AS2': (1.2966, 103.7764),
            'AS3': (1.2966, 103.7764),
            'AS4': (1.2966, 103.7764),
            'AS5': (1.2966, 103.7764),
            'AS6': (1.2966, 103.7764),
            'AS7': (1.2966, 103.7764),
            'AS8': (1.2966, 103.7764),
            
            # Business School (Kent Ridge Campus)
            'BIZ1': (1.2966, 103.7764),
            'BIZ2': (1.2966, 103.7764),
            'BIZ3': (1.2966, 103.7764),
            'BIZ4': (1.2966, 103.7764),
            'BIZ5': (1.2966, 103.7764),
            'BIZ6': (1.2966, 103.7764),
            'BIZ7': (1.2966, 103.7764),
            'BIZ8': (1.2966, 103.7764),
            
            # Computing (Kent Ridge Campus)
            'COM1': (1.2966, 103.7764),
            'COM2': (1.2966, 103.7764),
            'COM3': (1.2966, 103.7764),
            'COM4': (1.2966, 103.7764),
            'COM5': (1.2966, 103.7764),
            'COM6': (1.2966, 103.7764),
            'COM7': (1.2966, 103.7764),
            'COM8': (1.2966, 103.7764),
            
            # Design & Environment (Kent Ridge Campus)
            'DE1': (1.2966, 103.7764),
            'DE2': (1.2966, 103.7764),
            'DE3': (1.2966, 103.7764),
            'DE4': (1.2966, 103.7764),
            'DE5': (1.2966, 103.7764),
            'DE6': (1.2966, 103.7764),
            'DE7': (1.2966, 103.7764),
            'DE8': (1.2966, 103.7764),
            
            # Medicine (Kent Ridge Campus)
            'MD1': (1.2966, 103.7764),
            'MD2': (1.2966, 103.7764),
            'MD3': (1.2966, 103.7764),
            'MD4': (1.2966, 103.7764),
            'MD5': (1.2966, 103.7764),
            'MD6': (1.2966, 103.7764),
            'MD7': (1.2966, 103.7764),
            'MD8': (1.2966, 103.7764),
            
            # Law (Kent Ridge Campus)
            'LAW1': (1.2966, 103.7764),
            'LAW2': (1.2966, 103.7764),
            'LAW3': (1.2966, 103.7764),
            'LAW4': (1.2966, 103.7764),
            'LAW5': (1.2966, 103.7764),
            'LAW6': (1.2966, 103.7764),
            'LAW7': (1.2966, 103.7764),
            'LAW8': (1.2966, 103.7764),
            
            # Default coordinates for unknown venues (Kent Ridge Campus)
            'UNKNOWN': (1.2966, 103.7764)
        }
    
    def _get_venue_coordinates(self, venue: str) -> Tuple[float, float]:
        """Get coordinates for a venue, with fallback for unknown venues"""
        # Extract building code from venue string
        venue_code = venue.split()[0] if venue else 'UNKNOWN'
        
        # Try exact match first
        if venue_code in self.venue_coordinates:
            return self.venue_coordinates[venue_code]
        
        # Try partial matches for common patterns
        for known_venue, coords in self.venue_coordinates.items():
            if venue_code.startswith(known_venue) or known_venue.startswith(venue_code):
                return coords
        
        # Default fallback
        return self.venue_coordinates['UNKNOWN']
    
    def _calculate_distance(self, venue1: str, venue2: str) -> float:
        """Calculate walking distance between two venues using Haversine formula"""
        if venue1 == venue2:
            return 0.0
        
        coord1 = self._get_venue_coordinates(venue1)
        coord2 = self._get_venue_coordinates(venue2)
        
        import math
        
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # convert to radianss
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        r = 6371
        
        distance_km = c * r
        return distance_km * 1000  
    
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

                if module == 'NST2046':
                    print(f"\nNST2046 {lesson_type} classes:")
                    for class_no, (var, sessions) in class_vars[module][lesson_type].items():
                        for session in sessions:
                            print(f"  Class {class_no}: {session['day']} {session['startTime']}-{session['endTime']}")
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
        
        if self.preferences.get('minimizeTravel'):
            travel_penalty = self._add_minimize_travel_constraint(model, all_selected_lessons)
            if travel_penalty is not None:
                objective_terms.append(travel_penalty)
        
        if self.preferences.get('compactSchedule'):
            gap_penalty = self._add_compact_schedule_constraint(model, all_selected_lessons)
            if gap_penalty is not None:
                objective_terms.append(gap_penalty)
        
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
            penalty = model.NewIntVar(0, len(friday_vars) * 2, 'friday_penalty')
            model.Add(penalty == sum(friday_vars) * 2)
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
            penalty = model.NewIntVar(0, len(lunch_vars) * 2, 'lunch_penalty')
            model.Add(penalty == sum(lunch_vars) * 2)
            return penalty
        return None
    
    def _add_minimize_travel_constraint(self, model, all_selected_lessons):
        """Penalize schedules that require significant travel between venues"""
        # Group lessons by day to calculate travel distances
        daily_lessons = defaultdict(list)
        for var, session, module, lesson_type, class_no in all_selected_lessons:
            daily_lessons[session['day']].append((var, session))
        
        travel_penalties = []
        
        for day, lessons in daily_lessons.items():
            if len(lessons) < 2:
                continue  # No travel needed for single lesson days
            
            # Calculate travel distances between consecutive lessons
            for i in range(len(lessons) - 1):
                var1, session1 = lessons[i]
                var2, session2 = lessons[i + 1]
                
                # Only penalize if both lessons are selected
                venue1 = session1['venue']
                venue2 = session2['venue']
                distance = self._calculate_distance(venue1, venue2)
                
                # Create penalty variable for this travel segment
                # Scale penalty based on distance (penalty increases with distance)
                max_penalty = int(distance / 100)  # 1 penalty point per 100
                travel_penalty = model.NewIntVar(0, max_penalty, f'travel_{day}_{i}')
                
                # If both lessons are selected, apply the penalty
                model.Add(travel_penalty >= (var1 + var2 - 1) * max_penalty)
                travel_penalties.append(travel_penalty)
        
        if travel_penalties:
            total_penalty = model.NewIntVar(0, sum(p.Proto().domain[1] for p in travel_penalties), 'total_travel_penalty')
            model.Add(total_penalty == sum(travel_penalties))
            return total_penalty
        return None
    
    def _add_compact_schedule_constraint(self, model, all_selected_lessons):
        """Penalize gaps between classes to encourage compact schedules"""
        # Group lessons by day
        daily_lessons = defaultdict(list)
        for var, session, module, lesson_type, class_no in all_selected_lessons:
            daily_lessons[session['day']].append((var, session))
    
        gap_penalties = []
    
        for day, lessons in daily_lessons.items():
            if len(lessons) < 2:
                continue  # No gaps for single lesson days
        
            # Sort lessons by start time (deterministic ordering)
            lessons_sorted = sorted(lessons, key=lambda x: x[1]['startTime'])

            # DEBUG: Print all lessons on this day
            print(f"\n{day} lessons (sorted):")
            for var, session in lessons_sorted:
                print(f"  {session['startTime']}-{session['endTime']}")

            # Calculate gaps between ALL pairs of lessons (not just consecutive)
            for i in range(len(lessons)):
                for j in range(i + 1, len(lessons)):
                    var1, session1 = lessons[i]
                    var2, session2 = lessons[j]
        
                    # Calculate gap in 30-min slots
                    end_slot1 = self.time_to_slot(session1['endTime'])
                    start_slot2 = self.time_to_slot(session2['startTime'])
                    gap_slots = start_slot2 - end_slot1
        
                    if gap_slots > 0:  # Only if session2 is after session1
                        # Convert to hours and create penalty
                        gap_hours = gap_slots // 2
                        max_penalty = gap_hours * 5  # 5 penalty points per gap hour
            
                        gap_penalty = model.NewIntVar(0, max_penalty, f'gap_{day}_{i}_{j}')
            
                        # If both lessons are selected, apply the penalty
                        model.Add(gap_penalty >= (var1 + var2 - 1) * max_penalty)
                        gap_penalties.append(gap_penalty)
    
        if gap_penalties:
            total_penalty = model.NewIntVar(0, sum(p.Proto().domain[1] for p in gap_penalties), 'total_gap_penalty')
            model.Add(total_penalty == sum(gap_penalties))
            return total_penalty
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
            score -= morning_count * 1
        
        if self.preferences.get('freeFridays'):
            friday_count = sum(1 for s in schedule if s['day'] == 'Friday')
            score -= friday_count * 2
        
        if self.preferences.get('lunchBreak'):
            lunch_conflicts = self._get_lunch_conflicts(schedule)
            score -= len(lunch_conflicts) * 2
        
        if self.preferences.get('minimizeTravel'):
            total_travel_distance = self._calculate_total_travel_distance(schedule)
            # Deduct 1 point per 100m of travel distance
            travel_penalty = int(total_travel_distance / 100)
            score -= travel_penalty

        if self.preferences.get('compactSchedule'):
            total_gaps = self._calculate_total_gaps(schedule)
            score -= total_gaps * 5 # Each hour of gap costs 5 points
    
        return max(0,score)
    
    def _calculate_total_travel_distance(self, schedule: List[Dict]) -> float:
        """Calculate total travel distance for a schedule"""
        if len(schedule) < 2:
            return 0.0
        
        # Group lessons by day
        daily_lessons = defaultdict(list)
        for slot in schedule:
            daily_lessons[slot['day']].append(slot)
        
        total_distance = 0.0
        
        for day, lessons in daily_lessons.items():
            if len(lessons) < 2:
                continue
            
            # Sort lessons by time
            lessons.sort(key=lambda x: x['startTime'])
            
            # Calculate distance between consecutive lessons
            for i in range(len(lessons) - 1):
                venue1 = lessons[i]['venue']
                venue2 = lessons[i + 1]['venue']
                distance = self._calculate_distance(venue1, venue2)
                total_distance += distance
        
        return total_distance
    
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
        
        if self.preferences.get('minimizeTravel'):
            total_travel_distance = self._calculate_total_travel_distance(schedule)
            if total_travel_distance < 100:  # Less than 100m total travel
                tradeoffs.append('✓ Minimal travel between venues')
            elif total_travel_distance < 500:  # Less than 500m total travel
                tradeoffs.append(f'⚠ Moderate travel: {total_travel_distance:.0f}m between venues')
            else:  # More than 500m total travel
                tradeoffs.append(f'⚠ High travel: {total_travel_distance:.0f}m between venues')
        
        # Show gap information if compact schedule preferred
        if self.preferences.get('compactSchedule'):
            total_gaps = self._calculate_total_gaps(schedule)
            if total_gaps == 0:
                tradeoffs.append('✓ No gaps between classes (fully compact)')
            else:
                tradeoffs.append(f'⏰ {total_gaps} hour(s) of gaps between classes')

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

    def _calculate_total_gaps(self, schedule: List[Dict]) -> int:
        """Calculate total hours of gaps between classes"""
        # Group by day
        days_schedule = defaultdict(list)
        for slot in schedule:
            days_schedule[slot['day']].append(slot)
    
        total_gap_hours = 0
        for day, slots in days_schedule.items():
            # Sort by start time
            sorted_slots = sorted(slots, key=lambda x: x['startTime'])
        
            # Calculate gaps between consecutive classes
            for i in range(len(sorted_slots) - 1):
                end_time = self.time_to_slot(sorted_slots[i]['endTime'])
                start_time = self.time_to_slot(sorted_slots[i + 1]['startTime'])
            
                gap_slots = start_time - end_time
                # Only count gaps > 0 (30 min intervals)
                if gap_slots > 0:
                    # Convert slots to hours (2 slots = 1 hour)
                    gap_hours = gap_slots // 2
                    total_gap_hours += gap_hours
    
        return total_gap_hours

# ==================== LLM Chat Optimizer (OpenAI) ====================

def format_module_data_for_llm(module_data_dict: Dict, lessons_dict: Dict) -> str:
    """Format module timetable data as simple text"""
    text = ""
    for module_code, lessons in lessons_dict.items():
        title = module_data_dict[module_code].get('title', '')
        text += f"\n{module_code} - {title}:\n"
        for lesson_type, classes in lessons.items():
            text += f"  {lesson_type}:\n"
            for class_no, sessions in classes.items():
                text += f"    Class {class_no}:\n"
                for session in sessions:
                    text += f"      {session['day']} {session['startTime']}-{session['endTime']} @ {session['venue']}\n"
    return text


def chat_optimize_timetable(modules: List[str], user_message: str) -> Dict:
    """Use OpenAI GPT to understand user preferences and generate optimized timetable"""
    
    if not LLM_AVAILABLE or not openai_client:
        return {
            'error': 'LLM service not available. Please set OPENAI_API_KEY environment variable.'
        }
    
    # Fetch module data
    module_data_dict = {}
    lessons_dict = {}
    
    for module_code in modules:
        data = NUSModsAPI.get_module_data(module_code)
        if data:
            module_data_dict[module_code] = data
            lessons_dict[module_code] = NUSModsAPI.parse_lessons(data)
    
    if not lessons_dict:
        return {'error': 'Failed to load module data'}
    
    # Format module data
    module_info = format_module_data_for_llm(module_data_dict, lessons_dict)
    
    # Create prompt for GPT
    system_prompt = """You are an expert NUS timetable optimizer. 

Your task:
1. Understand the user's scheduling preferences
2. Analyze available lesson slots
3. Select ONE class for each lesson type (Lecture, Tutorial, Lab, etc.) for each module
4. Ensure no time conflicts
5. Return 2-3 optimal timetable options as JSON

Return ONLY valid JSON in this exact format:
{
  "timetables": [
    {
      "schedule": [
        {"module": "CS1010", "type": "Lecture", "classNo": "1", "day": "Monday", "startTime": "0900", "endTime": "1100", "venue": "COM1-0210", "weeks": [1,2,3,4,5,6,7,8,9,10,11,12,13]}
      ],
      "explanation": "Brief explanation why this timetable is good",
      "tradeoffs": ["✓ Pros", "⚠ Cons"]
    }
  ]
}"""

    user_prompt = f"""Create optimized timetables for: {', '.join(modules)}

User's request: "{user_message}"

Available lesson slots:
{module_info}

Generate 2-3 optimal timetable options. Return ONLY the JSON response."""

    try:
        print(f"\n🤖 Calling OpenAI GPT for: {modules}")
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        llm_response = response.choices[0].message.content.strip()
        
        # Clean markdown formatting
        if "```json" in llm_response:
            llm_response = llm_response.split("```json")[1].split("```")[0].strip()
        elif "```" in llm_response:
            llm_response = llm_response.split("```")[1].split("```")[0].strip()
        
        result = json.loads(llm_response)
        print(f"✓ Generated {len(result.get('timetables', []))} timetables\n")
        
        return {
            'success': True,
            'timetables': result.get('timetables', []),
            'count': len(result.get('timetables', []))
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return {'error': f'Failed to parse LLM response: {str(e)}'}
    except Exception as e:
        print(f"LLM error: {e}")
        traceback.print_exc()
        return {'error': f'LLM optimization failed: {str(e)}'}

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

@app.route('/api/chat-optimize', methods=['POST'])
def chat_optimize():
    """
    LLM chat-based optimization endpoint using OpenAI GPT
    
    Expected JSON body:
    {
        "modules": ["NST2046", "ST3131", "BT3102", "QF3101", "NHS2099", "IS4226"],
        "message": "I want to minimize travel and squeeze everything into one day"
    }
    """
    try:
        data = request.json
        modules = data.get('modules', [])
        user_message = data.get('message', '')
        
        print(f"\n{'='*60}")
        print(f"🤖 LLM Chat Optimization Request (OpenAI):")
        print(f"Modules: {modules}")
        print(f"Message: {user_message}")
        print(f"{'='*60}\n")
        
        if not modules:
            return jsonify({'error': 'No modules provided'}), 400
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Use LLM to optimize
        result = chat_optimize_timetable(modules, user_message)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NUSMods Timetable Optimizer Backend")
    print("="*60)
    print("Server starting on http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)