import React, { useState } from "react";
import {
  Calendar,
  Clock,
  MapPin,
  AlertCircle,
  CheckCircle,
  Loader,
} from "lucide-react";
import "./App.css";

// 🆕 Enhanced BlockedTimeForm Component
function BlockedTimeForm({ onAdd, onCancel }) {
  const [day, setDay] = useState('Monday');
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [applyToAllDays, setApplyToAllDays] = useState(false); // 🆕 Add this state

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // 🆕 If "Apply to all weekdays" is checked, add to all weekdays
    if (applyToAllDays) {
      const weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];
      weekdays.forEach(weekday => {
        onAdd(weekday, startTime, endTime);
      });
    } else {
      // Add to selected day only
      onAdd(day, startTime, endTime);
    }
    
    // Reset form
    setStartTime('');
    setEndTime('');
    setApplyToAllDays(false);
  };

  // 🆕 Quick preset buttons
  const applyPreset = (start, end) => {
    setStartTime(start);
    setEndTime(end);
  };

  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      
      {/* 🆕 Quick Presets Section */}
      <div>
        <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#374151', marginBottom: '10px' }}>
          ⚡ Quick Presets
        </label>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button
            type="button"
            onClick={() => applyPreset('1200', '1300')}
            style={{
              padding: '8px 16px',
              background: '#f3f4f6',
              color: '#374151',
              border: '2px solid #e5e7eb',
              borderRadius: '6px',
              fontSize: '13px',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = '#e5e7eb';
              e.currentTarget.style.borderColor = '#667eea';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = '#f3f4f6';
              e.currentTarget.style.borderColor = '#e5e7eb';
            }}
          >
            🍱 Lunch (12-1pm)
          </button>
          <button
            type="button"
            onClick={() => applyPreset('1800', '2000')}
            style={{
              padding: '8px 16px',
              background: '#f3f4f6',
              color: '#374151',
              border: '2px solid #e5e7eb',
              borderRadius: '6px',
              fontSize: '13px',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = '#e5e7eb';
              e.currentTarget.style.borderColor = '#667eea';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = '#f3f4f6';
              e.currentTarget.style.borderColor = '#e5e7eb';
            }}
          >
            🌙 Evening (6-8pm)
          </button>
          <button
            type="button"
            onClick={() => applyPreset('0900', '1700')}
            style={{
              padding: '8px 16px',
              background: '#f3f4f6',
              color: '#374151',
              border: '2px solid #e5e7eb',
              borderRadius: '6px',
              fontSize: '13px',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = '#e5e7eb';
              e.currentTarget.style.borderColor = '#667eea';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = '#f3f4f6';
              e.currentTarget.style.borderColor = '#e5e7eb';
            }}
          >
            💼 Work Hours (9am-5pm)
          </button>
          <button
            type="button"
            onClick={() => applyPreset('0800', '2200')}
            style={{
              padding: '8px 16px',
              background: '#f3f4f6',
              color: '#374151',
              border: '2px solid #e5e7eb',
              borderRadius: '6px',
              fontSize: '13px',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = '#e5e7eb';
              e.currentTarget.style.borderColor = '#667eea';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = '#f3f4f6';
              e.currentTarget.style.borderColor = '#e5e7eb';
            }}
          >
            📅 Full Day
          </button>
        </div>
      </div>

      {/* Time Input Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        
        {/* Day Selection - 🆕 Disabled when "Apply to all weekdays" is checked */}
        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#374151', marginBottom: '8px' }}>
            Day
          </label>
          <select
            value={day}
            onChange={(e) => setDay(e.target.value)}
            disabled={applyToAllDays} // 🆕 Disable when applying to all days
            style={{
              width: '100%',
              padding: '12px',
              border: '2px solid #e5e7eb',
              borderRadius: '8px',
              fontSize: '15px',
              outline: 'none',
              cursor: applyToAllDays ? 'not-allowed' : 'pointer',
              opacity: applyToAllDays ? 0.5 : 1, // 🆕 Visual feedback
              background: applyToAllDays ? '#f9fafb' : 'white'
            }}
          >
            {days.map(d => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>

        {/* Start Time - 🆕 Enhanced with validation feedback */}
        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#374151', marginBottom: '8px' }}>
            Start Time (HHMM)
          </label>
          <input
            type="text"
            value={startTime}
            onChange={(e) => {
              // 🆕 Only allow numbers
              const value = e.target.value.replace(/\D/g, '');
              setStartTime(value);
            }}
            placeholder="e.g., 0900"
            maxLength="4"
            required
            style={{
              width: '100%',
              padding: '12px',
              border: '2px solid #e5e7eb',
              borderRadius: '8px',
              fontSize: '15px',
              outline: 'none'
            }}
            onFocus={(e) => e.target.style.borderColor = '#667eea'}
            onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
          />
        </div>

        {/* End Time - 🆕 Enhanced with validation feedback */}
        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#374151', marginBottom: '8px' }}>
            End Time (HHMM)
          </label>
          <input
            type="text"
            value={endTime}
            onChange={(e) => {
              // 🆕 Only allow numbers
              const value = e.target.value.replace(/\D/g, '');
              setEndTime(value);
            }}
            placeholder="e.g., 1100"
            maxLength="4"
            required
            style={{
              width: '100%',
              padding: '12px',
              border: '2px solid #e5e7eb',
              borderRadius: '8px',
              fontSize: '15px',
              outline: 'none'
            }}
            onFocus={(e) => e.target.style.borderColor = '#667eea'}
            onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
          />
        </div>
      </div>

      {/* 🆕 Apply to All Weekdays Checkbox - ADD THIS HERE */}
      <label style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '12px', 
        cursor: 'pointer',
        padding: '14px',
        background: applyToAllDays ? '#ede9fe' : '#f9fafb',
        borderRadius: '10px',
        border: applyToAllDays ? '2px solid #a78bfa' : '2px solid #e5e7eb',
        transition: 'all 0.2s'
      }}>
        <input
          type="checkbox"
          checked={applyToAllDays}
          onChange={(e) => setApplyToAllDays(e.target.checked)}
          style={{ 
            width: '20px', 
            height: '20px', 
            cursor: 'pointer', 
            accentColor: '#667eea' 
          }}
        />
        <div style={{ flex: 1 }}>
          <span style={{ color: '#374151', fontSize: '15px', fontWeight: '600', display: 'block' }}>
            📆 Apply to all weekdays (Mon-Fri)
          </span>
          <span style={{ color: '#6b7280', fontSize: '13px' }}>
            Block this time slot across Monday to Friday
          </span>
        </div>
      </label>

      {/* 🆕 Visual Preview of what will be added */}
      {(startTime.length === 4 && endTime.length === 4) && (
        <div style={{
          padding: '12px 16px',
          background: '#f0fdf4',
          border: '1px solid #bbf7d0',
          borderRadius: '8px',
          fontSize: '14px',
          color: '#166534'
        }}>
          <strong>Preview:</strong> Will block{' '}
          {applyToAllDays ? 'Monday-Friday' : day}{' '}
          from {startTime.slice(0, 2)}:{startTime.slice(2, 4)} to {endTime.slice(0, 2)}:{endTime.slice(2, 4)}
        </div>
      )}

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
        <button
          type="button"
          onClick={onCancel}
          style={{
            padding: '12px 24px',
            background: '#6b7280',
            color: 'white',
            borderRadius: '8px',
            border: 'none',
            fontSize: '15px',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
          onMouseOver={(e) => e.currentTarget.style.background = '#4b5563'}
          onMouseOut={(e) => e.currentTarget.style.background = '#6b7280'}
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={startTime.length !== 4 || endTime.length !== 4} // 🆕 Disable if invalid
          style={{
            padding: '12px 24px',
            background: (startTime.length === 4 && endTime.length === 4) ? '#10b981' : '#9ca3af',
            color: 'white',
            borderRadius: '8px',
            border: 'none',
            fontSize: '15px',
            fontWeight: '600',
            cursor: (startTime.length === 4 && endTime.length === 4) ? 'pointer' : 'not-allowed',
            transition: 'all 0.2s'
          }}
          onMouseOver={(e) => {
            if (startTime.length === 4 && endTime.length === 4) {
              e.currentTarget.style.background = '#059669';
            }
          }}
          onMouseOut={(e) => {
            if (startTime.length === 4 && endTime.length === 4) {
              e.currentTarget.style.background = '#10b981';
            }
          }}
        >
          Add Blocked Time
        </button>
      </div>

      {/* Help Text */}
      <p style={{ fontSize: '13px', color: '#6b7280', margin: 0, textAlign: 'center' }}>
        💡 Tip: Use 24-hour format (e.g., 0900 for 9 AM, 1430 for 2:30 PM)
      </p>
    </form>
  );
}


function App() {
  // Add this useEffect to set body styles
  React.useEffect(() => {
    document.body.style.margin = "0";
    document.body.style.padding = "0";
    document.body.style.overflow = "hidden";
    document.documentElement.style.margin = "0";
    document.documentElement.style.padding = "0";
  }, []);

  // Helper function to format weeks display
  const formatWeeks = (weeks) => {
    if (!weeks || weeks.length === 0) return "All weeks";

    const sorted = [...weeks].sort((a, b) => a - b);
    const allOdd = sorted.every((w) => w % 2 === 1);
    const allEven = sorted.every((w) => w % 2 === 0);

    if (allOdd) return "Odd weeks";
    if (allEven) return "Even weeks";

    // For mixed or specific weeks, show compressed format
    if (sorted.length <= 3) {
      return `Wks ${sorted.join(", ")}`;
    }
    return `Wks ${sorted[0]}-${sorted[sorted.length - 1]}`;
  };

  const [modules, setModules] = useState("");
  const [preferences, setPreferences] = useState({
    noMorningClasses: false,
    compactSchedule: false,
    freeFridays: false,
    lunchBreak: true,
    minimizeTravel: false,
  });

  const [blockedTimes, setBlockedTimes] = useState([]);
  const [showBlockedTimeForm, setShowBlockedTimeForm] = useState(false);

  const [loading, setLoading] = useState(false);
  const [timetables, setTimetables] = useState([]);
  const [error, setError] = useState('');

  // Helper function to add a blocked time slot
  const addBlockedTime = (day, startTime, endTime) => {
    // Validate input
    if (!day || !startTime || !endTime) {
      alert('Please fill in all fields');
      return;
    }

    // Validate time format (should be HHMM)
    const timeRegex = /^([0-1][0-9]|2[0-3])[0-5][0-9]$/;
    if (!timeRegex.test(startTime) || !timeRegex.test(endTime)) {
      alert('Invalid time format. Use HHMM (e.g., 0900, 1430)');
      return;
    }

    // Validate start < end
    if (parseInt(startTime) >= parseInt(endTime)) {
      alert('Start time must be before end time');
      return;
    }

    const newBlockedTime = { day, startTime, endTime };
    setBlockedTimes([...blockedTimes, newBlockedTime]);
  };

  // Helper function to remove a blocked time slot
  const removeBlockedTime = (index) => {
    setBlockedTimes(blockedTimes.filter((_, i) => i !== index));
  };

  // Format time for display (0900 -> 09:00)
  const formatTimeDisplay = (time) => {
    return `${time.slice(0, 2)}:${time.slice(2, 4)}`;
  };

  // Function to call your backend API
const optimizeTimetable = async () => {
  setLoading(true);
  setError('');
  setTimetables([]);

    try {
      // Parse modules
      const moduleList = modules
        .split(",")
        .map((m) => m.trim().toUpperCase())
        .filter((m) => m);

      if (moduleList.length === 0) {
        throw new Error("Please enter at least one module code");
      }

    // Include blocked times in preferences
    const requestBody = { 
      modules: moduleList, 
      preferences: {
        ...preferences,
        blockedTimes: blockedTimes  // Include blocked times
      }
    };

      console.log("Sending request to backend:", {
        modules: moduleList,
        preferences,
      });

    // Call the Python backend API
    const response = await fetch('http://localhost:5001/api/optimize', {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(requestBody),
    });


      console.log("Response status:", response.status);

      const data = await response.json();
      console.log("Response data:", data);

      if (!response.ok) {
        throw new Error(data.error || "Failed to optimize timetable");
      }

      if (data.timetables && data.timetables.length > 0) {
        setTimetables(data.timetables);
        console.log("Timetables set successfully:", data.timetables);
      } else {
        throw new Error("No feasible timetables found");
      }
    } catch (err) {
      setError(err.message);
      console.error("Optimization error:", err);
    } finally {
      setLoading(false);
    }
  };

  const exportToNUSMods = (timetable) => {
    // Generate NUSMods share link format
    // Format: ?MOD_CODE[LESSON_TYPE]=CLASS_NO
    let shareParams = [];
    const moduleClasses = {};

    // Group by module
    timetable.schedule.forEach((slot) => {
      if (!moduleClasses[slot.module]) {
        moduleClasses[slot.module] = {};
      }
      moduleClasses[slot.module][slot.type] = slot.classNo;
    });

    // Build share link
    for (const [module, classes] of Object.entries(moduleClasses)) {
      for (const [type, classNo] of Object.entries(classes)) {
        shareParams.push(`${module}[${type}]=${classNo}`);
      }
    }

    const shareLink = `https://nusmods.com/timetable/sem-1?${shareParams.join(
      "&"
    )}`;
    window.open(shareLink, "_blank");
  };

  const exportToCalendar = (timetable) => {
    // Generate .ics file content
    let icsContent =
      "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//NUSMods Optimizer//EN\n";

    timetable.schedule.forEach((slot) => {
      icsContent += "BEGIN:VEVENT\n";
      icsContent += `SUMMARY:${slot.module} ${slot.type}\n`;
      icsContent += `LOCATION:${slot.venue}\n`;
      icsContent += `DESCRIPTION:Class ${slot.classNo}\n`;
      // Note: In production, you'd need to calculate actual dates and recurrence rules
      icsContent += "END:VEVENT\n";
    });

    icsContent += "END:VCALENDAR";

    // Create download
    const blob = new Blob([icsContent], { type: "text/calendar" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `timetable_${timetable.id}.ics`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100vw",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        padding: "40px 0",
        margin: 0,
        boxSizing: "border-box",
        overflowX: "hidden",
      }}
    >
      <div
        style={{ width: "100%", padding: "0 60px", boxSizing: "border-box" }}
      >
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: "60px" }}>
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "20px",
              marginBottom: "20px",
            }}
          >
            <Calendar size={56} color="white" />
            <h1
              style={{
                fontSize: "52px",
                fontWeight: "bold",
                color: "white",
                margin: 0,
                textShadow: "0 4px 6px rgba(0,0,0,0.2)",
              }}
            >
              NUSMods Timetable Optimizer
            </h1>
          </div>
          <p
            style={{
              color: "rgba(255,255,255,0.95)",
              fontSize: "20px",
              margin: 0,
              textShadow: "0 2px 4px rgba(0,0,0,0.1)",
            }}
          >
            AI-powered scheduling using Google OR-Tools and NUSMods API
          </p>
        </div>

        {/* Input Section */}
        <div
          style={{
            background: "white",
            borderRadius: "20px",
            boxShadow: "0 25px 70px rgba(0,0,0,0.4)",
            padding: "50px 60px",
            marginBottom: "40px",
          }}
        >
          <h2
            style={{
              fontSize: "28px",
              fontWeight: "700",
              marginBottom: "30px",
              color: "#1f2937",
              textAlign: "center",
            }}
          >
            📚 Module Selection
          </h2>

          <div style={{ marginBottom: "40px" }}>
            <label
              style={{
                display: "block",
                fontSize: "16px",
                fontWeight: "600",
                color: "#374151",
                marginBottom: "12px",
              }}
            >
              Enter Module Codes (comma-separated)
            </label>
            <input
              type="text"
              value={modules}
              onChange={(e) => setModules(e.target.value)}
              placeholder="e.g., CS2030S, CS2040S, MA1521, GEA1000"
              style={{
                width: "100%",
                padding: "16px 20px",
                border: "2px solid #e5e7eb",
                borderRadius: "12px",
                fontSize: "16px",
                outline: "none",
                transition: "all 0.3s ease",
              }}
              onFocus={(e) => (e.target.style.borderColor = "#667eea")}
              onBlur={(e) => (e.target.style.borderColor = "#e5e7eb")}
            />
            <p style={{ fontSize: "14px", color: "#6b7280", marginTop: "8px" }}>
              Example NUS modules for AY2025-2026
            </p>
          </div>

          <h3
            style={{
              fontSize: "24px",
              fontWeight: "700",
              marginBottom: "24px",
              color: "#1f2937",
              textAlign: "center",
            }}
          >
            ⚙️ Your Preferences
          </h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: "20px",
              marginBottom: "40px",
            }}
          >
            {[
              ["noMorningClasses", "No classes before 10am"],
              ["compactSchedule", "Compact schedule (minimize gaps)"],
              ["freeFridays", "Keep Fridays free"],
              ["lunchBreak", "Daily lunch break (12-2pm)"],
              ["minimizeTravel", "Minimize campus travel"],
            ].map(([key, label]) => (
              <label
                key={key}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "12px",
                  cursor: "pointer",
                  padding: "12px",
                  borderRadius: "8px",
                  transition: "background 0.2s",
                  background: preferences[key] ? "#f3f4f6" : "transparent",
                }}
              >
                <input
                  type="checkbox"
                  checked={preferences[key]}
                  onChange={(e) =>
                    setPreferences({ ...preferences, [key]: e.target.checked })
                  }
                  style={{
                    width: "20px",
                    height: "20px",
                    cursor: "pointer",
                    accentColor: "#667eea",
                  }}
                />
                <span
                  style={{
                    color: "#374151",
                    fontSize: "15px",
                    fontWeight: "500",
                  }}
                >
                  {label}
                </span>
              </label>
            ))}
          </div>

          <div style={{ marginTop: '40px', marginBottom: '30px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
              <h3 style={{ fontSize: '24px', fontWeight: '700', color: '#1f2937', margin: 0 }}>
                🚫 Block Time Slots
              </h3>
              <button
                onClick={() => setShowBlockedTimeForm(!showBlockedTimeForm)}
                style={{
                  padding: '10px 20px',
                  background: showBlockedTimeForm ? '#ef4444' : '#10b981',
                  color: 'white',
                  borderRadius: '8px',
                  border: 'none',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                {showBlockedTimeForm ? '✕ Cancel' : '+ Add Blocked Time'}
              </button>
            </div>

            {/* Add Blocked Time Form */}
            {showBlockedTimeForm && (
              <div style={{
                background: '#f9fafb',
                padding: '24px',
                borderRadius: '12px',
                border: '2px dashed #d1d5db',
                marginBottom: '20px'
              }}>
                <BlockedTimeForm onAdd={addBlockedTime} onCancel={() => setShowBlockedTimeForm(false)} />
              </div>
            )}

            {/* Display Current Blocked Times */}
            {blockedTimes.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {blockedTimes.map((blocked, index) => (
                  <div key={index} style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '16px 20px',
                    background: '#fef2f2',
                    border: '2px solid #fecaca',
                    borderRadius: '10px',
                    transition: 'all 0.2s'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                      <AlertCircle size={20} color="#dc2626" />
                      <span style={{ fontSize: '16px', fontWeight: '600', color: '#991b1b' }}>
                        {blocked.day}
                      </span>
                      <span style={{ fontSize: '15px', color: '#7f1d1d' }}>
                        {formatTimeDisplay(blocked.startTime)} - {formatTimeDisplay(blocked.endTime)}
                      </span>
                    </div>
                    <button
                      onClick={() => removeBlockedTime(index)}
                      style={{
                        padding: '8px 16px',
                        background: '#dc2626',
                        color: 'white',
                        borderRadius: '6px',
                        border: 'none',
                        fontSize: '14px',
                        fontWeight: '600',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                      }}
                      onMouseOver={(e) => e.currentTarget.style.background = '#b91c1c'}
                      onMouseOut={(e) => e.currentTarget.style.background = '#dc2626'}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}

            {blockedTimes.length === 0 && (
              <p style={{ fontSize: '14px', color: '#6b7280', textAlign: 'center', fontStyle: 'italic' }}>
                No blocked time slots. Add times when you're unavailable (e.g., club meetings, part-time work).
              </p>
            )}
          </div>

          <button
            onClick={optimizeTimetable}
            disabled={loading}
            style={{
              width: "100%",
              background: loading
                ? "#9ca3af"
                : "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              color: "white",
              padding: "16px",
              borderRadius: "12px",
              fontSize: "18px",
              fontWeight: "600",
              border: "none",
              cursor: loading ? "not-allowed" : "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "12px",
              boxShadow: loading
                ? "none"
                : "0 8px 20px rgba(102, 126, 234, 0.4)",
              transform: loading ? "none" : "translateY(0)",
              transition: "all 0.3s ease",
            }}
            onMouseOver={(e) => {
              if (!loading) {
                e.currentTarget.style.transform = "translateY(-2px)";
                e.currentTarget.style.boxShadow =
                  "0 12px 28px rgba(102, 126, 234, 0.5)";
              }
            }}
            onMouseOut={(e) => {
              if (!loading) {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow =
                  "0 8px 20px rgba(102, 126, 234, 0.4)";
              }
            }}
          >
            {loading ? (
              <>
                <Loader
                  size={20}
                  style={{ animation: "spin 1s linear infinite" }}
                />
                Optimizing with OR-Tools...
              </>
            ) : (
              "Generate Optimal Timetables"
            )}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div
            style={{
              background: "#fef2f2",
              border: "1px solid #fecaca",
              borderRadius: "8px",
              padding: "16px",
              marginBottom: "24px",
              display: "flex",
              alignItems: "start",
              gap: "12px",
            }}
          >
            <AlertCircle
              size={20}
              color="#dc2626"
              style={{ flexShrink: 0, marginTop: "2px" }}
            />
            <p style={{ color: "#991b1b", margin: 0 }}>{error}</p>
          </div>
        )}

        {/* Results Section */}
        {timetables.length > 0 && (
          <div
            style={{ display: "flex", flexDirection: "column", gap: "40px" }}
          >
            <h2
              style={{
                fontSize: "40px",
                fontWeight: "bold",
                color: "white",
                textAlign: "center",
                textShadow: "0 4px 8px rgba(0,0,0,0.3)",
              }}
            >
              ✨ Your Optimized Timetables
            </h2>

            {timetables.map((timetable) => (
              <div
                key={timetable.id}
                style={{
                  background: "white",
                  borderRadius: "20px",
                  boxShadow: "0 25px 70px rgba(0,0,0,0.4)",
                  padding: "40px",
                  border: "3px solid rgba(102, 126, 234, 0.3)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginBottom: "24px",
                    flexWrap: "wrap",
                    gap: "20px",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "16px",
                    }}
                  >
                    <h3
                      style={{
                        fontSize: "28px",
                        fontWeight: "700",
                        color: "#1f2937",
                        margin: 0,
                      }}
                    >
                      Option {timetable.id}
                    </h3>
                    <span
                      style={{
                        background:
                          timetable.score >= 90
                            ? "#dcfce7"
                            : timetable.score >= 70
                            ? "#fef3c7"
                            : "#fee2e2",
                        color:
                          timetable.score >= 90
                            ? "#166534"
                            : timetable.score >= 70
                            ? "#92400e"
                            : "#991b1b",
                        padding: "8px 16px",
                        borderRadius: "9999px",
                        fontSize: "16px",
                        fontWeight: "600",
                      }}
                    >
                      Score: {timetable.score}/100
                    </span>
                  </div>
                  <div
                    style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}
                  >
                    <button
                      onClick={() => exportToNUSMods(timetable)}
                      style={{
                        padding: "12px 24px",
                        background: "#2563eb",
                        color: "white",
                        borderRadius: "10px",
                        border: "none",
                        fontSize: "16px",
                        fontWeight: "600",
                        cursor: "pointer",
                        transition: "all 0.2s",
                        boxShadow: "0 4px 12px rgba(37, 99, 235, 0.3)",
                      }}
                      onMouseOver={(e) => {
                        e.currentTarget.style.background = "#1d4ed8";
                        e.currentTarget.style.transform = "translateY(-2px)";
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.background = "#2563eb";
                        e.currentTarget.style.transform = "translateY(0)";
                      }}
                    >
                      Open in NUSMods
                    </button>
                    <button
                      onClick={() => exportToCalendar(timetable)}
                      style={{
                        padding: "12px 24px",
                        background: "#9333ea",
                        color: "white",
                        borderRadius: "10px",
                        border: "none",
                        fontSize: "16px",
                        fontWeight: "600",
                        cursor: "pointer",
                        transition: "all 0.2s",
                        boxShadow: "0 4px 12px rgba(147, 51, 234, 0.3)",
                      }}
                      onMouseOver={(e) => {
                        e.currentTarget.style.background = "#7e22ce";
                        e.currentTarget.style.transform = "translateY(-2px)";
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.background = "#9333ea";
                        e.currentTarget.style.transform = "translateY(0)";
                      }}
                    >
                      Export .ics
                    </button>
                  </div>
                </div>

                {/* Schedule */}
                <div style={{ marginBottom: "24px" }}>
                  <h4
                    style={{
                      fontWeight: "700",
                      color: "#374151",
                      marginBottom: "16px",
                      fontSize: "20px",
                    }}
                  >
                    📅 Weekly Schedule
                  </h4>
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "12px",
                    }}
                  >
                    {timetable.schedule.map((slot, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "2fr 2fr 1fr 1fr",
                          gap: "20px",
                          padding: "18px 20px",
                          background: "#f9fafb",
                          borderRadius: "12px",
                          alignItems: "center",
                          border: "1px solid #e5e7eb",
                          transition: "all 0.2s",
                        }}
                        onMouseOver={(e) => {
                          e.currentTarget.style.background = "#f3f4f6";
                          e.currentTarget.style.borderColor = "#667eea";
                        }}
                        onMouseOut={(e) => {
                          e.currentTarget.style.background = "#f9fafb";
                          e.currentTarget.style.borderColor = "#e5e7eb";
                        }}
                      >
                        <div style={{ minWidth: "180px" }}>
                          <div
                            style={{
                              fontWeight: "700",
                              color: "#1f2937",
                              fontSize: "16px",
                            }}
                          >
                            {slot.module}
                          </div>
                          <div
                            style={{
                              fontSize: "14px",
                              color: "#6b7280",
                              marginTop: "2px",
                            }}
                          >
                            {slot.type}
                          </div>
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "10px",
                            fontSize: "15px",
                            color: "#4b5563",
                          }}
                        >
                          <Clock size={18} />
                          <span style={{ fontWeight: "500" }}>
                            {slot.day} {slot.startTime}-{slot.endTime}
                          </span>
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "10px",
                            fontSize: "15px",
                            color: "#4b5563",
                          }}
                        >
                          <MapPin size={18} />
                          <span style={{ fontWeight: "500" }}>
                            {slot.venue}
                          </span>
                        </div>
                        <div
                          style={{
                            fontSize: "14px",
                            color: "#6b7280",
                            fontWeight: "500",
                          }}
                        >
                          {formatWeeks(slot.weeks)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Tradeoffs */}
                <div>
                  <h4
                    style={{
                      fontWeight: "700",
                      color: "#374151",
                      marginBottom: "16px",
                      display: "flex",
                      alignItems: "center",
                      gap: "10px",
                      fontSize: "20px",
                    }}
                  >
                    <CheckCircle size={24} color="#16a34a" />
                    Trade-offs & Explanations
                  </h4>
                  <ul
                    style={{ margin: 0, paddingLeft: "0", listStyle: "none" }}
                  >
                    {timetable.tradeoffs.map((tradeoff, idx) => (
                      <li
                        key={idx}
                        style={{
                          color: "#374151",
                          marginBottom: "12px",
                          padding: "12px 16px",
                          background: "#f9fafb",
                          borderRadius: "8px",
                          fontSize: "15px",
                          borderLeft: "4px solid #667eea",
                        }}
                      >
                        {tradeoff}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Info Box */}
        <div
          style={{
            background: "rgba(255, 255, 255, 0.98)",
            borderRadius: "20px",
            padding: "32px",
            marginTop: "40px",
            backdropFilter: "blur(10px)",
            border: "2px solid rgba(255,255,255,0.5)",
            boxShadow: "0 10px 40px rgba(0,0,0,0.15)",
          }}
        >
          <h4
            style={{
              fontWeight: "700",
              color: "#1f2937",
              marginBottom: "20px",
              fontSize: "24px",
              textAlign: "center",
            }}
          >
            🚀 How it works
          </h4>
          <ul
            style={{
              fontSize: "16px",
              color: "#374151",
              margin: 0,
              paddingLeft: "28px",
              lineHeight: "2",
            }}
          >
            <li>Fetches module data from NUSMods API</li>
            <li>
              Uses Google OR-Tools CP-SAT solver for constraint satisfaction
            </li>
            <li>
              Generates multiple optimized timetables based on your preferences
            </li>
            <li>
              Explains trade-offs in plain language for informed decision-making
            </li>
            <li>Export to NUSMods or .ics calendar format</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default App;
