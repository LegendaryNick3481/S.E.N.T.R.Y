import React from 'react';

export function TradingSignals({ signals, loading }: { signals: any[], loading: boolean }) {
  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <h2>Trading Signals</h2>
      <ul>
        {signals.map((signal, index) => (
          <li key={index}>{JSON.stringify(signal)}</li>
        ))}
      </ul>
    </div>
  );
}