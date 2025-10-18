"""
Flask Backend API for NUSMods Timetable Optimizer
Run with: python backend.py
"""

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
    ACADEMIC_YEAR = "2024-2025"
    
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
    def parse_lessons(module_data: Dict) -> Dict[str, List[Dict]]:
        """Parse lessons into structured format by lesson type"""
        lessons_by_type = defaultdict(list)
        
        if not module_data or 'semesterData' not in module_data:
            return lessons_by_type
        
        # Get first semester data (Semester 1)
        semester_data = None
        for sem in module_data['semesterData']:
            if sem['semester'] == 1:
                semester_data = sem
                break
        
        if not semester_data or 'timetable' not in semester_data:
            return lessons_by_type
        
        for lesson in semester_data['timetable']:
            lesson_type = lesson['lessonType']
            lessons_by_type[lesson_type].append({
                'classNo': lesson['classNo'],
                'day': lesson['day'],
                'startTime': lesson['startTime'],
                'endTime': lesson['endTime'],
                'venue': lesson['venue'],
                'weeks': lesson.get('weeks', [])
            })
        
        return dict(lessons_by_type)


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
    
    @staticmethod
    def time_to_slot(time_str: str) -> int:
        """Convert time string (e.g., '0900') to slot index (30-min intervals from 8am)"""
        hours = int(time_str[:2])
        minutes = int(time_str[2:])
        return (hours - 8) * 2 + (minutes // 30)
    
    @staticmethod
    def slot_to_time(slot: int) -> str:
        """Convert slot index to time string"""
        hours = 8 + (slot // 2)
        minutes = (slot % 2) * 30
        return f"{hours:02d}{minutes:02d}"
    
    def optimize(self) -> List[Dict]:
        """Run optimization and return multiple timetable solutions"""
        self.fetch_module_data()
        
        # Check if we have valid modules
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
        
        # Decision variables: for each module and lesson type, pick one class
        class_vars = {}
        for module, lessons_by_type in self.lessons_by_module.items():
            if not lessons_by_type:
                continue
                
            class_vars[module] = {}
            for lesson_type, classes in lessons_by_type.items():
                class_vars[module][lesson_type] = {}
                for lesson in classes:
                    class_id = lesson['classNo']
                    var = model.NewBoolVar(f"{module}_{lesson_type}_{class_id}")
                    class_vars[module][lesson_type][class_id] = (var, lesson)
                
                # Constraint: exactly one class must be chosen per lesson type
                model.Add(sum(var for var, _ in class_vars[module][lesson_type].values()) == 1)
        
        if not class_vars:
            print("No valid class variables created")
            return []
        
        # Constraint: No overlapping classes
        self._add_no_overlap_constraints(model, class_vars)
        
         # 🆕 NEW: Constraint: Block user-specified time slots (HARD CONSTRAINT)
        self._add_blocked_time_constraints(model, class_vars)

        # Soft constraints (preferences) as objectives
        objective_terms = []
        
        if self.preferences.get('noMorningClasses'):
            morning_penalty = self._add_no_morning_classes(model, class_vars)
            if morning_penalty is not None:
                objective_terms.append(morning_penalty)
        
        if self.preferences.get('freeFridays'):
            friday_penalty = self._add_free_fridays(model, class_vars)
            if friday_penalty is not None:
                objective_terms.append(friday_penalty)
        
        if self.preferences.get('lunchBreak'):
            lunch_penalty = self._add_lunch_break_constraint(model, class_vars)
            if lunch_penalty is not None:
                objective_terms.append(lunch_penalty)
        
        if self.preferences.get('minimizeTravel'):
            travel_penalty = self._add_minimize_travel(model, class_vars)
            if travel_penalty is not None:
                objective_terms.append(travel_penalty)
        
        if self.preferences.get('compactSchedule'):
            gap_penalty = self._add_compact_schedule(model, class_vars)
            if gap_penalty is not None:
                objective_terms.append(gap_penalty)
        
        # Set objective: minimize penalties
        if objective_terms:
            model.Minimize(sum(objective_terms))
        
        # Solve and collect multiple solutions
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        
        solutions = []
        
        class SolutionCollector(cp_model.CpSolverSolutionCallback):
            def __init__(self, variables, limit):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self._variables = variables
                self._solution_limit = limit
                self.solutions = []
            
            def on_solution_callback(self):
                if len(self.solutions) >= self._solution_limit:
                    self.StopSearch()
                    return
                
                solution = {}
                for module, types in self._variables.items():
                    solution[module] = {}
                    for lesson_type, classes in types.items():
                        for class_id, (var, lesson) in classes.items():
                            if self.Value(var):
                                solution[module][lesson_type] = lesson
                                break
                self.solutions.append(solution)
        
        solution_collector = SolutionCollector(class_vars, limit=5)
        status = solver.Solve(model, solution_collector)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print(f"Found {len(solution_collector.solutions)} solution(s)")
            return self._format_solutions(solution_collector.solutions)
        else:
            print(f"No feasible solution found. Status: {status}")
            return []
    
    def _add_no_overlap_constraints(self, model, class_vars):
        """Ensure no two classes overlap in time"""
        all_classes = []
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    all_classes.append((var, lesson))
        
        for i in range(len(all_classes)):
            for j in range(i + 1, len(all_classes)):
                var1, lesson1 = all_classes[i]
                var2, lesson2 = all_classes[j]
                
                if self._lessons_overlap(lesson1, lesson2):
                    # If both are selected, they overlap - not allowed
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
    
    def _add_blocked_time_constraints(self, model, class_vars):
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
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    # Check if this lesson conflicts with any blocked time
                    for blocked in blocked_times:
                        if self._lesson_conflicts_with_blocked_time(lesson, blocked):
                            # This class cannot be selected (set to 0)
                            model.Add(var == 0)
                            blocked_count += 1
                            print(f"  ✗ Blocked: {module} {lesson_type} [{class_id}] on {lesson['day']} "
                                  f"{lesson['startTime']}-{lesson['endTime']} (conflicts with blocked time)")
                            break  # No need to check other blocked times for this lesson
    
        print(f"Total classes blocked: {blocked_count}")

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
        """Check if two lessons overlap"""
        if lesson1['day'] != lesson2['day']:
            return False
        
        start1 = self.time_to_slot(lesson1['startTime'])
        end1 = self.time_to_slot(lesson1['endTime'])
        start2 = self.time_to_slot(lesson2['startTime'])
        end2 = self.time_to_slot(lesson2['endTime'])
        
        return not (end1 <= start2 or end2 <= start1)
    
    def _add_no_morning_classes(self, model, class_vars):
        """Penalize classes before 10am"""
        morning_vars = []
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    start_slot = self.time_to_slot(lesson['startTime'])
                    if start_slot < 4:  # Before 10am (8am = 0, 10am = 4)
                        morning_vars.append(var)
        
        if morning_vars:
            penalty = model.NewIntVar(0, len(morning_vars) * 10, 'morning_penalty')
            model.Add(penalty == sum(morning_vars) * 10)
            return penalty
        return None
    
    def _add_free_fridays(self, model, class_vars):
        """Penalize Friday classes"""
        friday_vars = []
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    if lesson['day'] == 'Friday':
                        friday_vars.append(var)
        
        if friday_vars:
            penalty = model.NewIntVar(0, len(friday_vars) * 15, 'friday_penalty')
            model.Add(penalty == sum(friday_vars) * 15)
            return penalty
        return None
    
    def _add_lunch_break_constraint(self, model, class_vars):
        """Penalize classes that overlap with lunch time (12pm-2pm)"""
        lunch_overlap_vars = []
        
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    start_slot = self.time_to_slot(lesson['startTime'])
                    end_slot = self.time_to_slot(lesson['endTime'])
                    
                    # Lunch time: 12pm = slot 8, 2pm = slot 12
                    # Check if class overlaps with lunch (not completely before or after)
                    if not (end_slot <= 8 or start_slot >= 12):
                        lunch_overlap_vars.append(var)
        
        if lunch_overlap_vars:
            # Lower penalty (8) so it prefers lunch breaks but doesn't make solution infeasible
            penalty = model.NewIntVar(0, len(lunch_overlap_vars) * 8, 'lunch_penalty')
            model.Add(penalty == sum(lunch_overlap_vars) * 8)
            return penalty
        return None
    
    def _add_minimize_travel(self, model, class_vars):
        """Penalize classes that require traveling between distant locations on the same day"""
        # Define location groups (buildings that are close together)
        location_groups = {
            'COM': ['COM1', 'COM2', 'COM3', 'I3'],  # Computing buildings
            'LT': ['LT', 'AS'],  # Lecture theaters in Arts/Science
            'S': ['S17', 'S16', 'S14', 'S11'],  # Science buildings
            'E': ['E1', 'E2', 'E3', 'E4', 'E5', 'EA', 'EW'],  # Engineering
            'UTOWN': ['UT', 'RC4'],  # UTown area
        }
        
        def get_location_group(venue):
            """Get the location group for a venue"""
            if not venue:
                return 'UNKNOWN'
            # Extract building prefix (e.g., 'COM1' from 'COM1-B111')
            for group, buildings in location_groups.items():
                for building in buildings:
                    if venue.startswith(building):
                        return group
            return 'OTHER'
        
        # Group classes by day and check for travel between different areas
        travel_penalty_vars = []
        
        # Get all possible class combinations
        all_class_options = []
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    all_class_options.append((var, lesson))
        
        # Check pairs of classes on the same day
        for i in range(len(all_class_options)):
            for j in range(i + 1, len(all_class_options)):
                var1, lesson1 = all_class_options[i]
                var2, lesson2 = all_class_options[j]
                
                # Only check if they're on the same day
                if lesson1['day'] == lesson2['day']:
                    loc1 = get_location_group(lesson1['venue'])
                    loc2 = get_location_group(lesson2['venue'])
                    
                    # If they're in different location groups, add penalty
                    if loc1 != loc2 and loc1 != 'OTHER' and loc2 != 'OTHER':
                        # Create a variable that's 1 if both classes are selected
                        travel_var = model.NewBoolVar(f'travel_{i}_{j}')
                        # travel_var is 1 only if both var1 and var2 are selected
                        model.Add(var1 + var2 == 2).OnlyEnforceIf(travel_var)
                        model.Add(var1 + var2 < 2).OnlyEnforceIf(travel_var.Not())
                        travel_penalty_vars.append(travel_var)
        
        if travel_penalty_vars:
            penalty = model.NewIntVar(0, len(travel_penalty_vars) * 5, 'travel_penalty')
            model.Add(penalty == sum(travel_penalty_vars) * 5)
            return penalty
        return None
    
    def _add_compact_schedule(self, model, class_vars):
        """Penalize large gaps between classes on the same day"""
        gap_penalty_vars = []
        
        # Get all possible class combinations
        all_class_options = []
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    all_class_options.append((var, lesson))
        
        # Check pairs of classes on the same day for gaps
        for i in range(len(all_class_options)):
            for j in range(i + 1, len(all_class_options)):
                var1, lesson1 = all_class_options[i]
                var2, lesson2 = all_class_options[j]
                
                # Only check if they're on the same day
                if lesson1['day'] == lesson2['day']:
                    # Calculate gap between classes
                    end1 = self.time_to_slot(lesson1['endTime'])
                    start2 = self.time_to_slot(lesson2['startTime'])
                    
                    # Ensure lesson1 ends before lesson2 starts
                    if end1 <= start2:
                        gap_slots = start2 - end1
                        # Penalize gaps > 2 hours (4 slots = 2 hours)
                        if gap_slots > 4:
                            # Create variable for when both classes are selected
                            gap_var = model.NewBoolVar(f'gap_{i}_{j}')
                            model.Add(var1 + var2 == 2).OnlyEnforceIf(gap_var)
                            model.Add(var1 + var2 < 2).OnlyEnforceIf(gap_var.Not())
                            # Penalty increases with gap size
                            gap_penalty_vars.extend([gap_var] * (gap_slots - 4))
        
        if gap_penalty_vars:
            penalty = model.NewIntVar(0, len(gap_penalty_vars) * 2, 'gap_penalty')
            model.Add(penalty == sum(gap_penalty_vars) * 2)
            return penalty
        return None
    
    def _format_solutions(self, raw_solutions: List[Dict]) -> List[Dict]:
        """Format solutions into readable timetables"""
        formatted = []
        for idx, solution in enumerate(raw_solutions):
            schedule = []
            for module, lessons in solution.items():
                for lesson_type, lesson_data in lessons.items():
                    schedule.append({
                        'module': module,
                        'type': lesson_type,
                        'classNo': lesson_data['classNo'],
                        'day': lesson_data['day'],
                        'startTime': lesson_data['startTime'],
                        'endTime': lesson_data['endTime'],
                        'venue': lesson_data['venue']
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
        
        # Deduct for morning classes
        if self.preferences.get('noMorningClasses'):
            morning_count = sum(1 for s in schedule if self.time_to_slot(s['startTime']) < 4)
            score -= morning_count * 5
            print(f"  Morning classes: {morning_count}, Score after: {score}")
        
        # Deduct for Friday classes
        if self.preferences.get('freeFridays'):
            friday_count = sum(1 for s in schedule if s['day'] == 'Friday')
            score -= friday_count * 10
            print(f"  Friday classes: {friday_count}, Score after: {score}")
        
        # Deduct for lunch break conflicts
        if self.preferences.get('lunchBreak'):
            lunch_conflicts = self._get_lunch_conflicts(schedule)
            conflict_count = len(lunch_conflicts)
            score -= conflict_count * 8  # 8 points per day with lunch conflict
            print(f"  Lunch conflicts on {conflict_count} day(s): {lunch_conflicts}, Score after: {score}")
        
        # Deduct for campus travel
        if self.preferences.get('minimizeTravel'):
            travel_count = self._count_campus_travel(schedule)
            score -= travel_count * 3  # 3 points per cross-campus trip
            print(f"  Cross-campus trips: {travel_count}, Score after: {score}")
        
        # Deduct for gaps in schedule
        if self.preferences.get('compactSchedule'):
            gap_hours = self._calculate_total_gaps(schedule)
            score -= gap_hours * 2  # 2 points per hour of gaps
            print(f"  Total gap hours: {gap_hours}, Score after: {score}")
        
        print(f"  Final score: {score}")
        return max(0, score)
    
    def _generate_tradeoffs(self, schedule: List[Dict]) -> List[str]:
        """Generate human-readable tradeoff explanations"""
        tradeoffs = []
        
        # Check for morning classes
        morning_classes = [s for s in schedule if self.time_to_slot(s['startTime']) < 4]
        if self.preferences.get('noMorningClasses'):
            if not morning_classes:
                tradeoffs.append('✓ No classes before 10am')
            else:
                tradeoffs.append(f'⚠ {len(morning_classes)} morning class(es) scheduled (before 10am)')
        
        # Check for Friday classes
        friday_classes = [s for s in schedule if s['day'] == 'Friday']
        if self.preferences.get('freeFridays'):
            if not friday_classes:
                tradeoffs.append('✓ Fridays are completely free')
            else:
                tradeoffs.append(f'⚠ {len(friday_classes)} class(es) on Friday')
        
        # Check for lunch breaks - detailed by day
        if self.preferences.get('lunchBreak'):
            lunch_conflicts = self._get_lunch_conflicts(schedule)
            if not lunch_conflicts:
                tradeoffs.append('✓ Lunch breaks (12-2pm) preserved on all days')
            else:
                days_str = ', '.join(lunch_conflicts)
                tradeoffs.append(f'⚠ Lunch break conflicts on: {days_str}')
        
        # Compact schedule analysis
        days_with_classes = set(s['day'] for s in schedule)
        tradeoffs.append(f'📅 Classes spread across {len(days_with_classes)} day(s)')
        
        # Show gap information if compact schedule preferred
        if self.preferences.get('compactSchedule'):
            total_gaps = self._calculate_total_gaps(schedule)
            if total_gaps == 0:
                tradeoffs.append('✓ No gaps between classes (fully compact)')
            else:
                tradeoffs.append(f'⏰ {total_gaps} hour(s) of gaps between classes')
        
        return tradeoffs
    
    def _get_lunch_conflicts(self, schedule: List[Dict]) -> List[str]:
        """Get list of days with lunch time conflicts"""
        days_schedule = defaultdict(list)
        for slot in schedule:
            days_schedule[slot['day']].append(slot)
        
        conflict_days = []
        for day, slots in days_schedule.items():
            for slot in slots:
                start = self.time_to_slot(slot['startTime'])
                end = self.time_to_slot(slot['endTime'])
                # Check if class overlaps with lunch (12pm = slot 8, 2pm = slot 12)
                if not (end <= 8 or start >= 12):
                    conflict_days.append(day)
                    break  # Only count each day once
        
        return conflict_days
    
    def _count_campus_travel(self, schedule: List[Dict]) -> int:
        """Count number of cross-campus trips in the schedule"""
        location_groups = {
            'COM': ['COM1', 'COM2', 'COM3', 'I3'],
            'LT': ['LT', 'AS'],
            'S': ['S17', 'S16', 'S14', 'S11'],
            'E': ['E1', 'E2', 'E3', 'E4', 'E5', 'EA', 'EW'],
            'UTOWN': ['UT', 'RC4'],
        }
        
        def get_location_group(venue):
            if not venue:
                return 'UNKNOWN'
            for group, buildings in location_groups.items():
                for building in buildings:
                    if venue.startswith(building):
                        return group
            return 'OTHER'
        
        # Group by day
        days_schedule = defaultdict(list)
        for slot in schedule:
            days_schedule[slot['day']].append(slot)
        
        travel_count = 0
        for day, slots in days_schedule.items():
            # Sort by time
            sorted_slots = sorted(slots, key=lambda x: x['startTime'])
            # Count transitions between different location groups
            for i in range(len(sorted_slots) - 1):
                loc1 = get_location_group(sorted_slots[i]['venue'])
                loc2 = get_location_group(sorted_slots[i + 1]['venue'])
                if loc1 != loc2 and loc1 != 'OTHER' and loc2 != 'OTHER':
                    travel_count += 1
        
        return travel_count
    
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
    
    def _check_lunch_break(self, schedule: List[Dict]) -> bool:
        """Check if there's a lunch break from 12-2pm on all days"""
        days_schedule = defaultdict(list)
        for slot in schedule:
            days_schedule[slot['day']].append(slot)
        
        for day, slots in days_schedule.items():
            for slot in slots:
                start = self.time_to_slot(slot['startTime'])
                end = self.time_to_slot(slot['endTime'])
                # Check if class overlaps with lunch (12pm = slot 8, 2pm = slot 12)
                if not (end <= 8 or start >= 12):
                    return False
        return True


# ==================== Flask API Routes ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Backend is running'})


@app.route('/api/optimize', methods=['POST'])
def optimize_timetable():
    """Main optimization endpoint"""
    """
    Main optimization endpoint
    
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
        
        # Run optimization
        optimizer = TimetableOptimizer(modules, preferences)
        timetables = optimizer.optimize()
        
        if not timetables:
            return jsonify({
                'error': 'No feasible timetables found. This might be due to:\n'
                         '- Module codes not available for current semester\n'
                         '- Too many conflicting constraints\n'
                         '- Invalid module codes'
            }), 404
        
        return jsonify({
            'success': True,
            'timetables': timetables,
            'count': len(timetables)
        })
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/modules/<module_code>', methods=['GET'])
def get_module_info(module_code):
    """Get information about a specific module"""
    try:
        data = NUSModsAPI.get_module_data(module_code)
        if not data:
            return jsonify({'error': f'Module {module_code} not found'}), 404
        
        lessons = NUSModsAPI.parse_lessons(data)
        
        return jsonify({
            'code': module_code,
            'title': data.get('title'),
            'description': data.get('description'),
            'lessons': lessons
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NUSMods Timetable Optimizer Backend")
    print("="*60)
    print("Server starting on http://localhost:5001")
    print("API Endpoints:")
    print("  - GET  /api/health")
    print("  - POST /api/optimize")
    print("  - GET  /api/modules/<module_code>")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)