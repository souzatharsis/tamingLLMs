import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const MemoryUsageChart = () => {
  const data = [
    { name: 'F16', value: 141.1 },
    { name: 'Q8_0', value: 75.0 },
    { name: 'Q6_K', value: 59.9 },
    { name: 'Q5_K_M', value: 49.9 },
    { name: 'Q4_K_M', value: 42.5 },
    { name: 'Q3_K_M', value: 34.3 },
    { name: 'Q2_K', value: 26.4 }
  ];

  return (
    <div className="w-full h-96 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="name"
            tick={{ fontSize: 12, fontWeight: 'bold' }}
          />
          <YAxis 
            label={{ 
              value: 'Model Size (GB)', 
              angle: -90, 
              position: 'insideLeft',
              style: { 
                textAnchor: 'middle',
                fontWeight: 'bold'
              }
            }}
            tick={{ fontSize: 12, fontWeight: 'bold' }}
          />
          <Tooltip 
            formatter={(value) => [`${value} GB`, 'Model Size']}
            contentStyle={{ 
              backgroundColor: '#fff', 
              border: '1px solid #ccc',
              fontWeight: 'bold'
            }}
          />
          <Line 
            type="monotone"
            dataKey="value" 
            stroke="#3eaf7c"
            strokeWidth={2}
            dot={{ fill: '#3eaf7c', r: 4 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MemoryUsageChart;