import { useEffect } from 'react'
import { settleConfirm, useConfirm } from '../lib/confirm'

/** The host's own confirmation (lib/confirm.ts). */
export function ConfirmDialog() {
  const req = useConfirm()
  useEffect(() => {
    if (!req) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') settleConfirm(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [req])
  if (!req) return null
  return (
    <div className="modal-back" onPointerDown={e => { if (e.target === e.currentTarget) settleConfirm(false) }}>
      <div className="modal confirm-modal" role="alertdialog" aria-modal="true" aria-labelledby={`confirm-${req.id}`}>
        <div className="modal-h" id={`confirm-${req.id}`}>{req.title}</div>
        <p className="confirm-msg">{req.message}</p>
        <div className="modal-f">
          <button className="btn" onClick={() => settleConfirm(false)}>取消</button>
          <button className="btn dark" autoFocus onClick={() => settleConfirm(true)}>{req.confirmLabel}</button>
        </div>
      </div>
    </div>
  )
}
