/////////////////////////////////////////////////////////////////////////////
// Name:        thumbnailctrl.cpp
// Purpose:     Displays a scrolling window of thumbnails
// Author:      Julian Smart
// Modified by: Anil Kumar
// Created:     03/08/04 17:22:46
// RCS-ID:      
// Copyright:   (c) Julian Smart
// Licence:     wxWidgets Licence
/////////////////////////////////////////////////////////////////////////////

#if defined(__GNUG__) && !defined(__APPLE__)
#pragma implementation "thumbnailctrl.h"
#endif

// For compilers that support precompilation, includes "wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP

#include "wx/wx.h"

#endif

#include "thumbnailctrl.h"

#include "wx/settings.h"
#include "wx/arrimpl.cpp"
#include "wx/image.h"
#include "wx/filename.h"
#include "wx/dcbuffer.h"
#include <algorithm>

WX_DEFINE_OBJARRAY(wxThumbnailItemArray);

wxDEFINE_EVENT(wxEVT_COMMAND_THUMBNAIL_SELECTION_CHANGED, wxThumbnailEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_THUMBNAIL_ITEM_SELECTED, wxThumbnailEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_THUMBNAIL_ITEM_DESELECTED, wxThumbnailEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_THUMBNAIL_LEFT_CLICK, wxThumbnailEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_THUMBNAIL_RIGHT_CLICK, wxThumbnailEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_THUMBNAIL_VIEW_RIGHT_CLICK, wxThumbnailEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_THUMBNAIL_LEFT_DCLICK, wxThumbnailEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_THUMBNAIL_RETURN, wxThumbnailEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_THUMBNAIL_DRAG_START, wxThumbnailEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_THUMBNAIL_SORTED, wxThumbnailEvent);

wxDEFINE_EVENT(wxEVT_COMMAND_THUMBNAIL_ITEM_HOVER_CHANGED, wxThumbnailEvent);

IMPLEMENT_CLASS(wxThumbnailCtrl, wxScrolledWindow)

IMPLEMENT_CLASS(wxThumbnailEvent, wxNotifyEvent)

BEGIN_EVENT_TABLE(wxThumbnailCtrl, wxScrolledWindow)
                EVT_MOUSE_EVENTS(wxThumbnailCtrl::OnMouse)
                EVT_MOTION(wxThumbnailCtrl::OnMouseMotion)
                EVT_LEAVE_WINDOW(wxThumbnailCtrl::OnMouseLeave)
                EVT_CHAR(wxThumbnailCtrl::OnChar)
                EVT_SIZE(wxThumbnailCtrl::OnSize)
                EVT_SET_FOCUS(wxThumbnailCtrl::OnSetFocus)
                EVT_KILL_FOCUS(wxThumbnailCtrl::OnKillFocus)

                EVT_MENU(wxID_SELECTALL, wxThumbnailCtrl::OnSelectAll)
                EVT_UPDATE_UI(wxID_SELECTALL, wxThumbnailCtrl::OnUpdateSelectAll)
END_EVENT_TABLE()

wxThumbnailCtrl *wxThumbnailCtrl::sm_currentThumbnailCtrl = nullptr;

/*!
 * wxThumbnailCtrl
 */

wxThumbnailCtrl::wxThumbnailCtrl() {
    Init();
}

wxThumbnailCtrl::wxThumbnailCtrl(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, long style) {
    Init();
    Create(parent, id, pos, size, style);
}

/// Creation
bool wxThumbnailCtrl::Create(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, long style) {
    if (!wxScrolledCanvas::Create(parent, id, pos, size, style | wxFULL_REPAINT_ON_RESIZE))
        return false;

    if (!GetFont().Ok()) {
        SetFont(wxSystemSettings::GetFont(wxSYS_DEFAULT_GUI_FONT));
    }
    CalculateOverallThumbnailSize();

    SetBackgroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_3DFACE));
    // m_tagBitmap = wxBitmap(tick_xpm);

    SetBackgroundStyle(wxBG_STYLE_CUSTOM);

    // Tell the sizers to use the given or best size    
    // SetBestFittingSize(size);
    SetInitialSize(size);

    return true;
}

/// Member initialisation
void wxThumbnailCtrl::Init() {
    m_thumbnailOverallSize = wxTHUMBNAIL_DEFAULT_OVERALL_SIZE;
    m_thumbnailImageSize = wxTHUMBNAIL_DEFAULT_IMAGE_SIZE;
    m_spacing = wxTHUMBNAIL_DEFAULT_SPACING;
    m_thumbnailMargin = wxTHUMBNAIL_DEFAULT_MARGIN;
    m_firstSelection = -1;
    m_lastSelection = -1;
    m_focussedThumbnailBackgroundColour = wxTHUMBNAIL_DEFAULT_FOCUSSED_BACKGROUND;
    m_unfocussedThumbnailBackgroundColour = wxTHUMBNAIL_DEFAULT_UNFOCUSSED_BACKGROUND;
    m_unselectedThumbnailBackgroundColour = wxTHUMBNAIL_DEFAULT_UNSELECTED_BACKGROUND;
    m_typeColour = wxTHUMBNAIL_DEFAULT_TYPE_COLOUR;
    m_tagColour = wxTHUMBNAIL_DEFAULT_TAG_COLOUR;
    m_focusRectColour = wxTHUMBNAIL_DEFAULT_FOCUS_RECT_COLOUR;
    m_focusItem = -1;
    m_showOutlines = false;
}

/// Append a single item
int wxThumbnailCtrl::Append(wxThumbnailItem *item) {
    int sz = (int) GetCount();
    m_items.Add(item);
    m_firstSelection = -1;
    m_lastSelection = -1;
    m_focusItem = -1;

    if (!IsFrozen()) {
        SetupScrollbars();
        Refresh();
    }
    return sz;
}

/// Insert a single item
int wxThumbnailCtrl::Insert(wxThumbnailItem *item, int pos) {
    m_items.Insert(item, pos);
    m_firstSelection = -1;
    m_lastSelection = -1;
    m_focusItem = -1;

    // Must now change selection indices because
    // items above it have moved up
    size_t i;
    for (i = 0; i < m_selections.GetCount(); i++) {
        if (m_selections[i] >= pos)
            m_selections[i] = m_selections[i] + 1;
    }
    // Ditto for tags
    for (i = 0; i < m_tags.GetCount(); i++) {
        if (m_tags[i] >= pos)
            m_tags[i] = m_tags[i] + 1;
    }

    if (!IsFrozen()) {
        SetupScrollbars();
        Refresh();
    }
    return pos;
}

/// Clear all items
void wxThumbnailCtrl::Clear() {
    m_firstSelection = -1;
    m_lastSelection = -1;
    m_focusItem = -1;
    m_items.Clear();
    m_selections.Clear();
    m_tags.Clear();
    m_hoverItem = wxNOT_FOUND;

    if (!IsFrozen()) {
        SetupScrollbars();
        Refresh();
    }
}

int wxThumbnailCtrl::Compare(wxThumbnailItem **item1, wxThumbnailItem **item2) {
    wxFAIL_MSG("Override wxThumbnailCtrl::Compare function and provide your implementaion.");
    return 0;
}

/// Sorts items
void wxThumbnailCtrl::Sort() {
    // preserve and restore selections & tags
    size_t i;
    size_t len = m_items.GetCount();
    for (i = 0; i < len; i++) {
        auto &item = m_items[i];
        int state = 0;
        if (IsSelected(i))
            state |= wxTHUMBNAIL_SELECTED;
        if (IsTagged(i))
            state |= wxTHUMBNAIL_TAGGED;
        item.SetState(state);
    }
    m_selections.Clear();
    m_tags.Clear();
    m_firstSelection = -1;
    m_lastSelection = -1;
    m_focusItem = -1;

    sm_currentThumbnailCtrl = this;

    m_items.Sort([](wxThumbnailItem **item1, wxThumbnailItem **item2) {
        return sm_currentThumbnailCtrl->Compare(item1, item2);
    });

    sm_currentThumbnailCtrl = NULL;

    wxThumbnailEvent cmdEvent(
            wxEVT_COMMAND_THUMBNAIL_SORTED,
            GetId());
    cmdEvent.SetEventObject(this);
    GetEventHandler()->ProcessEvent(cmdEvent);

    Freeze();

    for (i = 0; i < len; i++) {
        auto &item = m_items[i];
        if (item.GetState() & wxTHUMBNAIL_SELECTED)
            Select(i);
        if (item.GetState() & wxTHUMBNAIL_TAGGED)
            Tag(i);
    }

    Thaw();
}

/// Delete this item
void wxThumbnailCtrl::Delete(int n) {
    if (m_firstSelection == n)
        m_firstSelection = -1;
    if (m_lastSelection == n)
        m_lastSelection = -1;
    if (m_focusItem == n)
        m_focusItem = -1;

    if (m_selections.Index(n) != wxNOT_FOUND) {
        m_selections.Remove(n);

        wxThumbnailEvent event(wxEVT_COMMAND_THUMBNAIL_ITEM_DESELECTED, GetId());
        event.SetEventObject(this);
        event.SetIndex(n);
        GetEventHandler()->ProcessEvent(event);

        wxThumbnailEvent cmdEvent(wxEVT_COMMAND_THUMBNAIL_SELECTION_CHANGED, GetId());
        cmdEvent.SetEventObject(this);
        GetEventHandler()->ProcessEvent(cmdEvent);
    }

    if (m_tags.Index(n) != wxNOT_FOUND)
        m_tags.Remove(n);

    m_items.RemoveAt(n);

    // Must now change selection indices because
    // items have moved down
    size_t i;
    for (i = 0; i < m_selections.GetCount(); i++) {
        if (m_selections[i] > n)
            m_selections[i] = m_selections[i] - 1;
    }

    if (!IsFrozen()) {
        SetupScrollbars();
        Refresh();
    }
}

/// Get the nth item
wxThumbnailItem *wxThumbnailCtrl::GetItem(int n) {
    wxASSERT(n < GetCount());

    if (n < GetCount()) {
        return &m_items[(size_t) n];
    } else {
        return nullptr;
    }
}

/// Get the overall rect of the given item
bool wxThumbnailCtrl::GetItemRect(int n, wxRect &rect, bool transform) {
    wxASSERT(n < GetCount());
    if (n < GetCount()) {
        int row, col;
        if (!GetRowCol(n, GetClientSize(), row, col))
            return false;

        int x = col * (m_thumbnailOverallSize.x + m_spacing) + m_spacing;
        int y = row * (m_thumbnailOverallSize.y + m_spacing) + m_spacing;

        if (transform) {
            int startX, startY;
            int xppu, yppu;
            GetScrollPixelsPerUnit(&xppu, &yppu);
            GetViewStart(&startX, &startY);
            x = x - startX * xppu;
            y = y - startY * yppu;
        }

        rect.x = x;
        rect.y = y;
        rect.width = m_thumbnailOverallSize.x;
        rect.height = m_thumbnailOverallSize.y;

        return true;
    }

    return false;
}

/// Get the image rect of the given item
bool wxThumbnailCtrl::GetItemRectImage(int n, wxRect &rect, bool transform) {
    wxASSERT(n < GetCount());

    wxRect outerRect;
    if (!GetItemRect(n, outerRect, transform))
        return false;

    rect.width = m_thumbnailImageSize.x;
    rect.height = m_thumbnailImageSize.y;
    rect.x = outerRect.x + (outerRect.width - rect.width) / 2;
    rect.y = outerRect.y + (outerRect.height - rect.height) / 2;
    rect.y -= m_thumbnailTextHeight / 2;

    return true;
}

/// The size of the image part
void wxThumbnailCtrl::SetThumbnailImageSize(const wxSize &sz) {
    m_thumbnailImageSize = sz;
    CalculateOverallThumbnailSize();

    if (GetCount() > 0 && !IsFrozen()) {
        SetupScrollbars();
        Refresh();
    }
}

/// Calculate the outer thumbnail size based
/// on font used for text and inner size
void wxThumbnailCtrl::CalculateOverallThumbnailSize() {
    wxCoord w;
    wxClientDC dc(this);
    dc.SetFont(GetFont());
    dc.GetTextExtent(wxT("X"), &w, &m_thumbnailTextHeight);

    // From left to right: margin, image, margin
    m_thumbnailOverallSize.x = m_thumbnailMargin * 2 + m_thumbnailImageSize.x;

    // From top to bottom: margin, image, margin, text, margin
    m_thumbnailOverallSize.y = m_thumbnailMargin * 3 + m_thumbnailTextHeight + m_thumbnailImageSize.y;

    SetMinClientSize(m_thumbnailOverallSize);
}

/// Return the row and column given the client
/// size and a left-to-right, top-to-bottom layout
/// assumption
bool wxThumbnailCtrl::GetRowCol(int item, const wxSize &clientSize, int &row, int &col) {
    wxASSERT(item < GetCount());
    if (item >= GetCount())
        return false;

    // How many can we fit in a row?

    int perRow = clientSize.x / (m_thumbnailOverallSize.x + m_spacing);
    if (perRow < 1)
        perRow = 1;

    row = item / perRow;
    col = item % perRow;

    return true;
}


/// Select or deselect an item
void wxThumbnailCtrl::Select(int n, bool select) {
    wxASSERT(n < GetCount());

    if (select) {
        if (m_selections.Index(n) == wxNOT_FOUND)
            m_selections.Add(n);
    } else {
        if (m_selections.Index(n) != wxNOT_FOUND)
            m_selections.Remove(n);
    }

    m_firstSelection = n;
    m_lastSelection = n;
    int oldFocusItem = m_focusItem;
    m_focusItem = n;

    if (!IsFrozen()) {
        wxRect rect;
        GetItemRect(n, rect);
        RefreshRect(rect);

        if (oldFocusItem != -1 && oldFocusItem != n) {
            GetItemRect(oldFocusItem, rect);
            RefreshRect(rect);
        }
    }
}

/// Select or deselect a range
void wxThumbnailCtrl::SelectRange(int from, int to, bool select) {
    int first = from;
    int last = to;
    if (first < last) {
        first = to;
        last = from;
    }
    wxASSERT(first >= 0 && first < GetCount());
    wxASSERT(last >= 0 && last < GetCount());

    Freeze();
    int i;
    for (i = first; i < last; i++) {
        Select(i, select);
    }
    m_focusItem = to;
    Thaw();
}

/// Select all
void wxThumbnailCtrl::SelectAll() {
    Freeze();
    int i;
    for (i = 0; i < GetCount(); i++) {
        Select(i, true);
    }
    if (GetCount() > 0) {
        m_focusItem = GetCount() - 1;
    } else {
        m_focusItem = -1;
    }
    Thaw();
}

/// Select none
void wxThumbnailCtrl::SelectNone() {
    Freeze();
    int i;
    for (i = 0; i < GetCount(); i++) {
        Select(i, false);
    }
    Thaw();
}

/// Get the index of the single selection, if not multi-select.
/// Returns -1 if there is no selection.
int wxThumbnailCtrl::GetSelection() const {
    if (m_selections.GetCount() > 0)
        return m_selections[0u];
    else
        return -1;
}

/// Returns true if the item is selected
bool wxThumbnailCtrl::IsSelected(int n) const {
    return (m_selections.Index(n) != wxNOT_FOUND);
}

/// Clears all selections
void wxThumbnailCtrl::ClearSelections() {
    int count = GetCount();

    m_selections.Clear();
    m_firstSelection = -1;
    m_lastSelection = -1;
    m_focusItem = -1;

    if (count > 0 && !IsFrozen()) {
        Refresh();
    }
}

/// Set the focus item
void wxThumbnailCtrl::SetFocusItem(int item) {
    wxASSERT(item < GetCount());
    if (item < GetCount()) {
        int oldFocusItem = m_focusItem;
        m_focusItem = item;

        if (!IsFrozen()) {
            wxRect rect;
            if (oldFocusItem != -1) {
                GetItemRect(oldFocusItem, rect);
                RefreshRect(rect);
            }
            if (m_focusItem != -1) {
                GetItemRect(m_focusItem, rect);
                RefreshRect(rect);
            }
        }
    }
}

/// Tag or untag an item
void wxThumbnailCtrl::Tag(int n, bool tag) {
    wxASSERT(n < GetCount());

    if (tag) {
        if (m_tags.Index(n) == wxNOT_FOUND)
            m_tags.Add(n);
    } else {
        if (m_tags.Index(n) != wxNOT_FOUND)
            m_tags.Remove(n);
    }

    if (!IsFrozen()) {
        wxRect rect;
        GetItemRect(n, rect);
        RefreshRect(rect);
    }
}

/// Returns true if the item is tagged
bool wxThumbnailCtrl::IsTagged(int n) const {
    return (m_tags.Index(n) != wxNOT_FOUND);
}

/// Clears all tags
void wxThumbnailCtrl::ClearTags() {
    int count = GetCount();

    m_tags.Clear();

    if (count > 0 && !IsFrozen()) {
        Refresh();
    }
}

/// Painting
void wxThumbnailCtrl::OnDraw(wxDC &dc) {
    if (IsFrozen())
        return;

    // Paint the background
    PaintBackground(dc);

    if (GetCount() == 0)
        return;

    wxRegion dirtyRegion = GetUpdateRegion();
    bool isFocussed = (FindFocus() == this);

    int i;
    int count = GetCount();
    int style = 0;
    wxRect rect, untransformedRect, imageRect, untransformedImageRect;
    for (i = 0; i < count; i++) {
        GetItemRect(i, rect);

        wxRegionContain c = dirtyRegion.Contains(rect);
        if (c != wxOutRegion) {
            GetItemRectImage(i, imageRect);
            style = 0;

            if (i == GetMouseHoverItem())
                style |= wxTHUMBNAIL_IS_HOVER;
            if (IsSelected(i))
                style |= wxTHUMBNAIL_SELECTED;
            if (IsTagged(i))
                style |= wxTHUMBNAIL_TAGGED;
            if (isFocussed)
                style |= wxTHUMBNAIL_FOCUSSED;
            if (isFocussed && i == m_focusItem)
                style |= wxTHUMBNAIL_IS_FOCUS;

            GetItemRect(i, untransformedRect, false);
            GetItemRectImage(i, untransformedImageRect, false);

            DrawItem(i, dc, untransformedRect, untransformedImageRect, style);
        }
    }
}

void wxThumbnailCtrl::OnSetFocus(wxFocusEvent & WXUNUSED(event)) {
    if (GetCount() > 0)
        Refresh();
}

void wxThumbnailCtrl::OnKillFocus(wxFocusEvent & WXUNUSED(event)) {
    if (GetCount() > 0)
        Refresh();
}

/// Mouse-event
void wxThumbnailCtrl::OnMouse(wxMouseEvent &event) {
    if (event.GetEventType() == wxEVT_MOUSEWHEEL) {
        // let the base handle mouse wheel events.
        event.Skip();
        return;
    }

    if (event.LeftDown()) {
        OnLeftClickDown(event);
    } else if (event.LeftUp()) {
        OnLeftClickUp(event);
    } else if (event.LeftDClick()) {
        OnLeftDClick(event);
    } else if (event.RightDown()) {
        OnRightClickDown(event);
    } else if (event.RightUp()) {
        OnRightClickUp(event);
    } else if (event.Dragging() && event.LeftIsDown() && GetSelections().Count()) {
        wxThumbnailEvent cmdEvent(
                wxEVT_COMMAND_THUMBNAIL_DRAG_START,
                GetId());

        if (wxDefaultPosition == m_dragStartPosition)
            m_dragStartPosition = event.GetPosition();

        if ((GetWindowStyle() & wxTH_MULTIPLE_SELECT) != 0)
            cmdEvent.SetItemsIndex(GetSelections());
        else
            cmdEvent.SetIndex(GetSelection());

        cmdEvent.SetPosition(m_dragStartPosition);
        cmdEvent.SetEventObject(this);
        GetEventHandler()->ProcessEvent(cmdEvent);
    } else {
        m_dragStartPosition = wxDefaultPosition;
    }

    event.Skip();
}

/// Left-click-down
void wxThumbnailCtrl::OnLeftClickDown(wxMouseEvent &event) {
    SetFocus();
    int n;
    if (HitTest(event.GetPosition(), n)) {
        int flags = 0;
        if (event.ControlDown())
            flags |= wxTHUMBNAIL_CTRL_DOWN;
        if (event.ShiftDown())
            flags |= wxTHUMBNAIL_SHIFT_DOWN;
        if (event.AltDown())
            flags |= wxTHUMBNAIL_ALT_DOWN;

        EnsureVisible(n);

        bool change = false;

        if (((GetWindowStyle() & wxTH_MULTIPLE_SELECT) != 0))
            change = !IsSelected(n) || (flags != 0);
        else
            change = !IsSelected(n);

        if (change)
            DoSelection(n, flags);

        wxThumbnailEvent cmdEvent(
                wxEVT_COMMAND_THUMBNAIL_LEFT_CLICK,
                GetId());
        cmdEvent.SetEventObject(this);
        cmdEvent.SetIndex(n);
        cmdEvent.SetFlags(flags);
        GetEventHandler()->ProcessEvent(cmdEvent);
    }
}

/// Left-click-up
void wxThumbnailCtrl::OnLeftClickUp(wxMouseEvent &event) {
    SetFocus();
    int n;
    if (HitTest(event.GetPosition(), n)) {
        int flags = 0;
        if (event.ControlDown())
            flags |= wxTHUMBNAIL_CTRL_DOWN;
        if (event.ShiftDown())
            flags |= wxTHUMBNAIL_SHIFT_DOWN;
        if (event.AltDown())
            flags |= wxTHUMBNAIL_ALT_DOWN;

        EnsureVisible(n);

        if ((GetWindowStyle() & wxTH_MULTIPLE_SELECT) != 0) {
            if ((GetSelections().Count() > 1) && IsSelected(n) && flags == 0)
                DoSelection(n, flags);
        }
    }
}

/// Right-click-down
void wxThumbnailCtrl::OnRightClickDown(wxMouseEvent &event) {
    SetFocus();
    int n;
    if (HitTest(event.GetPosition(), n)) {
        int flags = 0;
        if (event.ControlDown())
            flags |= wxTHUMBNAIL_CTRL_DOWN;
        if (event.ShiftDown())
            flags |= wxTHUMBNAIL_SHIFT_DOWN;
        if (event.AltDown())
            flags |= wxTHUMBNAIL_ALT_DOWN;

        if (m_focusItem != n)
            SetFocusItem(n);

        const wxArrayInt &selections = GetSelections();
        if (std::find(selections.begin(), selections.end(), n) == selections.end()) {
            SelectNone();
            Select(n);
            wxThumbnailEvent cmdEvent(
                    wxEVT_COMMAND_THUMBNAIL_ITEM_SELECTED,
                    GetId());
            cmdEvent.SetEventObject(this);
            cmdEvent.SetIndex(n);
            cmdEvent.SetFlags(flags);
            GetEventHandler()->ProcessEvent(cmdEvent);

            wxThumbnailEvent event(wxEVT_COMMAND_THUMBNAIL_SELECTION_CHANGED, GetId());
            event.SetEventObject(this);
            GetEventHandler()->ProcessEvent(event);
        }
    }

    event.Skip();
}

/// Right-click-up
void wxThumbnailCtrl::OnRightClickUp(wxMouseEvent &event) {
    SetFocus();

    int flags = 0;
    if (event.ControlDown())
        flags |= wxTHUMBNAIL_CTRL_DOWN;
    if (event.ShiftDown())
        flags |= wxTHUMBNAIL_SHIFT_DOWN;
    if (event.AltDown())
        flags |= wxTHUMBNAIL_ALT_DOWN;

    int n;
    if (HitTest(event.GetPosition(), n)) {
        if (m_focusItem != n)
            SetFocusItem(n);

        const wxArrayInt &selections = GetSelections();
        if (std::find(selections.begin(), selections.end(), n) != selections.end()) {
            wxThumbnailEvent cmdEvent(
                    wxEVT_COMMAND_THUMBNAIL_RIGHT_CLICK,
                    GetId());
            cmdEvent.SetEventObject(this);
            cmdEvent.SetIndex(n);
            cmdEvent.SetFlags(flags);
            cmdEvent.SetPosition(event.GetPosition());
            GetEventHandler()->ProcessEvent(cmdEvent);
        }
    } else {
        wxThumbnailEvent cmdEvent(
                wxEVT_COMMAND_THUMBNAIL_VIEW_RIGHT_CLICK,
                GetId());
        cmdEvent.SetEventObject(this);
        cmdEvent.SetFlags(flags);
        cmdEvent.SetPosition(event.GetPosition());
        GetEventHandler()->ProcessEvent(cmdEvent);
    }
}

/// Left-double-click
void wxThumbnailCtrl::OnLeftDClick(wxMouseEvent &event) {
    int n;
    if (HitTest(event.GetPosition(), n)) {
        int flags = 0;
        if (event.ControlDown())
            flags |= wxTHUMBNAIL_CTRL_DOWN;
        if (event.ShiftDown())
            flags |= wxTHUMBNAIL_SHIFT_DOWN;
        if (event.AltDown())
            flags |= wxTHUMBNAIL_ALT_DOWN;

        wxThumbnailEvent cmdEvent(
                wxEVT_COMMAND_THUMBNAIL_LEFT_DCLICK,
                GetId());
        cmdEvent.SetEventObject(this);
        cmdEvent.SetIndex(n);
        cmdEvent.SetFlags(flags);
        GetEventHandler()->ProcessEvent(cmdEvent);
    }
}

/// Mouse motion
void wxThumbnailCtrl::OnMouseMotion(wxMouseEvent &event) {
    int flags = 0;
    if (event.ControlDown())
        flags |= wxTHUMBNAIL_CTRL_DOWN;
    if (event.ShiftDown())
        flags |= wxTHUMBNAIL_SHIFT_DOWN;
    if (event.AltDown())
        flags |= wxTHUMBNAIL_ALT_DOWN;

    int n;
    if (HitTest(event.GetPosition(), n)) {
        SetMouseHoverItem(n, flags);
        ShowTooltip(n);
    } else {
        SetMouseHoverItem(wxNOT_FOUND, flags);
        UnsetToolTip();
    }

    event.Skip();
}

/// Mouse leave
void wxThumbnailCtrl::OnMouseLeave(wxMouseEvent &event) {
    int flags = 0;
    if (event.ControlDown())
        flags |= wxTHUMBNAIL_CTRL_DOWN;
    if (event.ShiftDown())
        flags |= wxTHUMBNAIL_SHIFT_DOWN;
    if (event.AltDown())
        flags |= wxTHUMBNAIL_ALT_DOWN;
    SetMouseHoverItem(wxNOT_FOUND, flags);

    event.Skip();
}

/// Key press
void wxThumbnailCtrl::OnChar(wxKeyEvent &event) {
    int flags = 0;
    if (event.ControlDown())
        flags |= wxTHUMBNAIL_CTRL_DOWN;
    if (event.ShiftDown())
        flags |= wxTHUMBNAIL_SHIFT_DOWN;
    if (event.AltDown())
        flags |= wxTHUMBNAIL_ALT_DOWN;

    if (event.GetKeyCode() == WXK_LEFT ||
        event.GetKeyCode() == WXK_RIGHT ||
        event.GetKeyCode() == WXK_UP ||
        event.GetKeyCode() == WXK_DOWN ||
        event.GetKeyCode() == WXK_HOME ||
        event.GetKeyCode() == WXK_PAGEUP ||
        event.GetKeyCode() == WXK_PAGEDOWN ||
        //event.GetKeyCode() == WXK_PRIOR ||
        //event.GetKeyCode() == WXK_NEXT ||
        event.GetKeyCode() == WXK_END) {
        Navigate(event.GetKeyCode(), flags);
    } else if (event.GetKeyCode() == WXK_RETURN) {
        wxThumbnailEvent cmdEvent(
                wxEVT_COMMAND_THUMBNAIL_RETURN,
                GetId());
        cmdEvent.SetEventObject(this);
        cmdEvent.SetFlags(flags);
        GetEventHandler()->ProcessEvent(cmdEvent);
    } else
        event.Skip();
}

/// Keyboard navigation
bool wxThumbnailCtrl::Navigate(int keyCode, int flags) {
    if (GetCount() == 0)
        return false;

    wxSize clientSize = GetClientSize();
    int perRow = clientSize.x / (m_thumbnailOverallSize.x + m_spacing);
    if (perRow < 1)
        perRow = 1;

    int rowsInView = clientSize.y / (m_thumbnailOverallSize.y + m_spacing);
    if (rowsInView < 1)
        rowsInView = 1;

    int focus = m_focusItem;
    if (focus == -1)
        focus = m_lastSelection;

    if (focus == -1 || focus >= GetCount()) {
        m_lastSelection = 0;
        DoSelection(m_lastSelection, flags);
        ScrollIntoView(m_lastSelection, keyCode);
        return true;
    }

    if (keyCode == WXK_RIGHT) {
        int next = focus + 1;
        if (next < GetCount()) {
            DoSelection(next, flags);
            ScrollIntoView(next, keyCode);
        }
    } else if (keyCode == WXK_LEFT) {
        int next = focus - 1;
        if (next >= 0) {
            DoSelection(next, flags);
            ScrollIntoView(next, keyCode);
        }
    } else if (keyCode == WXK_UP) {
        int next = focus - perRow;
        if (next >= 0) {
            DoSelection(next, flags);
            ScrollIntoView(next, keyCode);
        }
    } else if (keyCode == WXK_DOWN) {
        int next = focus + perRow;
        if (next < GetCount()) {
            DoSelection(next, flags);
            ScrollIntoView(next, keyCode);
        }
    } else if (keyCode == WXK_PAGEUP /*|| keyCode == WXK_PRIOR*/) {
        int next = focus - (perRow * rowsInView);
        if (next < 0)
            next = 0;
        if (next >= 0) {
            DoSelection(next, flags);
            ScrollIntoView(next, keyCode);
        }
    } else if (keyCode == WXK_PAGEDOWN /*|| keyCode == WXK_NEXT*/) {
        int next = focus + (perRow * rowsInView);
        if (next >= GetCount())
            next = GetCount() - 1;
        if (next < GetCount()) {
            DoSelection(next, flags);
            ScrollIntoView(next, keyCode);
        }
    } else if (keyCode == WXK_HOME) {
        DoSelection(0, flags);
        ScrollIntoView(0, keyCode);
    } else if (keyCode == WXK_END) {
        DoSelection(GetCount() - 1, flags);
        ScrollIntoView(GetCount() - 1, keyCode);
    }
    return true;
}

/// Scroll to see the image
void wxThumbnailCtrl::ScrollIntoView(int n, int keyCode) {
    wxRect rect;
    GetItemRect(n, rect, false); // _Not_ relative to scroll start

    int ppuX, ppuY;
    GetScrollPixelsPerUnit(&ppuX, &ppuY);

    int startX, startY;
    GetViewStart(&startX, &startY);
    startX = 0;
    startY = startY * ppuY;

    int sx, sy;
    GetVirtualSize(&sx, &sy);
    sx = 0;
    if (ppuY != 0)
        sy = sy / ppuY;

    wxSize clientSize = GetClientSize();

    // Going down
    if (keyCode == WXK_DOWN || keyCode == WXK_RIGHT || keyCode == WXK_END /*|| keyCode == WXK_NEXT*/ ||
        keyCode == WXK_PAGEDOWN) {
        if ((rect.y + rect.height) > (clientSize.y + startY)) {
            // Make it scroll so this item is at the bottom
            // of the window
            int y = rect.y - (clientSize.y - m_thumbnailOverallSize.y - m_spacing);
            SetScrollbars(ppuX, ppuY, sx, sy, 0, (int) (0.5 + y / ppuY));
        } else if (rect.y < startY) {
            // Make it scroll so this item is at the top
            // of the window
            int y = rect.y;
            SetScrollbars(ppuX, ppuY, sx, sy, 0, (int) (0.5 + y / ppuY));
        }
    }
        // Going up
    else if (keyCode == WXK_UP || keyCode == WXK_LEFT || keyCode == WXK_HOME /*|| keyCode == WXK_PRIOR*/ ||
             keyCode == WXK_PAGEUP) {
        if (rect.y < startY) {
            // Make it scroll so this item is at the top
            // of the window
            int y = rect.y;
            SetScrollbars(ppuX, ppuY, sx, sy, 0, (int) (0.5 + y / ppuY));
        } else if ((rect.y + rect.height) > (clientSize.y + startY)) {
            // Make it scroll so this item is at the bottom
            // of the window
            int y = rect.y - (clientSize.y - m_thumbnailOverallSize.y - m_spacing);
            SetScrollbars(ppuX, ppuY, sx, sy, 0, (int) (0.5 + y / ppuY));
        }
    }
}

/// Scrolls the item into view if necessary
void wxThumbnailCtrl::EnsureVisible(int n) {
    wxRect rect;
    GetItemRect(n, rect, false); // _Not_ relative to scroll start

    int ppuX, ppuY;
    GetScrollPixelsPerUnit(&ppuX, &ppuY);

    if (ppuY == 0)
        return;

    int startX, startY;
    GetViewStart(&startX, &startY);
    startX = 0;
    startY = startY * ppuY;

    int sx, sy;
    GetVirtualSize(&sx, &sy);
    sx = 0;
    if (ppuY != 0)
        sy = sy / ppuY;

    wxSize clientSize = GetClientSize();

    if ((rect.y + rect.height) > (clientSize.y + startY)) {
        // Make it scroll so this item is at the bottom
        // of the window
        int y = rect.y - (clientSize.y - m_thumbnailOverallSize.y - m_spacing);
        SetScrollbars(ppuX, ppuY, sx, sy, 0, (int) (0.5 + y / ppuY));
    } else if (rect.y < startY) {
        // Make it scroll so this item is at the top
        // of the window
        int y = rect.y;
        SetScrollbars(ppuX, ppuY, sx, sy, 0, (int) (0.5 + y / ppuY));
    }
}

/// Sizing
void wxThumbnailCtrl::OnSize(wxSizeEvent &event) {
    SetupScrollbars();
    event.Skip();
}

/// Set up scrollbars, e.g. after a resize
void wxThumbnailCtrl::SetupScrollbars() {
    if (IsFrozen())
        return;

    if (GetCount() == 0) {
        SetScrollbars(0, 0, 0, 0, 0, 0);
        return;
    }

    int lastItem = wxMax(0, GetCount() - 1);
    int pixelsPerUnit = 10;
    wxSize clientSize = GetClientSize();

    int row, col;
    GetRowCol(lastItem, clientSize, row, col);

    int maxHeight = (row + 1) * (m_thumbnailOverallSize.y + m_spacing) + m_spacing;

    int unitsY = maxHeight / pixelsPerUnit;

    int startX, startY;
    GetViewStart(&startX, &startY);

    int maxPositionX = 0; // wxMax(sz.x - clientSize.x, 0);
    int maxPositionY = (wxMax(maxHeight - clientSize.y, 0)) / pixelsPerUnit;

    // Move to previous scroll position if
    // possible
    SetScrollbars(0, pixelsPerUnit,
                  0, unitsY,
                  wxMin(maxPositionX, startX), wxMin(maxPositionY, startY));
}

/// Show the tooltip
void wxThumbnailCtrl::ShowTooltip(int n) {
    if (wxNOT_FOUND != n) {
        auto *item = GetItem(n);
        const wxString tooltip = item->GetLabel();

        if (GetToolTipText() != tooltip)
            SetToolTip(tooltip);
    }
}

/// Draws the background for the item, including bevel
bool wxThumbnailCtrl::DrawItem(int n, wxDC &dc, const wxRect &rect, const wxRect &imageRect, int style) {
    auto item = GetItem(n);
    if (item) {
        return item->DrawBackground(dc, this, rect, imageRect, style, n) && item->Draw(dc, this, imageRect, style, n);
    } else {
        return false;
    }
}

/// Do (de)selection
void wxThumbnailCtrl::DoSelection(int n, int flags) {
    bool isSelected = IsSelected(n);

    wxArrayInt stateChanged;

    bool multiSelect = (GetWindowStyle() & wxTH_MULTIPLE_SELECT) != 0;

    if (multiSelect && (flags & wxTHUMBNAIL_CTRL_DOWN) == wxTHUMBNAIL_CTRL_DOWN) {
        Select(n, !isSelected);
        stateChanged.Add(n);
    } else if (multiSelect && (flags & wxTHUMBNAIL_SHIFT_DOWN) == wxTHUMBNAIL_SHIFT_DOWN) {
        // We need to find the last item selected,
        // and select all in between.

        int first = m_firstSelection;

        // Want to keep the 'first' selection
        // if we're extending the selection
        bool keepFirstSelection = false;
        wxArrayInt oldSelections = m_selections;

        m_selections.Clear(); // TODO: need to refresh those that become unselected. Store old selections, compare with new

        if (m_firstSelection != -1 && m_firstSelection < GetCount() && m_firstSelection != n) {
            int step = (n < m_firstSelection) ? -1 : 1;
            int i;
            for (i = m_firstSelection; i != n; i += step) {
                if (!IsSelected(i)) {
                    m_selections.Add(i);
                    stateChanged.Add(i);

                    wxRect rect;
                    GetItemRect(i, rect);
                    RefreshRect(rect);
                }
            }
            keepFirstSelection = true;
        }

        // Refresh all the previously selected items that became unselected
        size_t i;
        for (i = 0; i < oldSelections.GetCount(); i++) {
            if (!IsSelected(oldSelections[i])) {
                wxRect rect;
                GetItemRect(oldSelections[i], rect);
                RefreshRect(rect);
            }
        }

        Select(n, true);
        if (stateChanged.Index(n) == wxNOT_FOUND)
            stateChanged.Add(n);

        if (keepFirstSelection)
            m_firstSelection = first;
    } else {
        size_t i = 0;
        for (i = 0; i < m_selections.GetCount(); i++) {
            wxRect rect;
            GetItemRect(m_selections[i], rect);
            RefreshRect(rect);

            stateChanged.Add(i);
        }

        m_selections.Clear();
        Select(n, true);
        if (stateChanged.Index(n) == wxNOT_FOUND)
            stateChanged.Add(n);
    }

    // Now notify the app of any selection changes
    size_t i = 0;
    for (i = 0; i < stateChanged.GetCount(); i++) {
        wxThumbnailEvent event(
                m_selections.Index(stateChanged[i]) != wxNOT_FOUND ? wxEVT_COMMAND_THUMBNAIL_ITEM_SELECTED
                                                                   : wxEVT_COMMAND_THUMBNAIL_ITEM_DESELECTED,
                GetId());
        event.SetEventObject(this);
        event.SetIndex(stateChanged[i]);
        GetEventHandler()->ProcessEvent(event);
    }

    if (stateChanged.GetCount() > 0) {
        wxThumbnailEvent event(wxEVT_COMMAND_THUMBNAIL_SELECTION_CHANGED, GetId());
        event.SetEventObject(this);
        GetEventHandler()->ProcessEvent(event);
    }
}

/// Find the item under the given point
bool wxThumbnailCtrl::HitTest(const wxPoint &pt, int &n) {
    wxSize clientSize = GetClientSize();
    int startX, startY;
    int ppuX, ppuY;
    GetViewStart(&startX, &startY);
    GetScrollPixelsPerUnit(&ppuX, &ppuY);

    int perRow = clientSize.x / (m_thumbnailOverallSize.x + m_spacing);
    if (perRow < 1)
        perRow = 1;

    int colPos = (int) (pt.x / (m_thumbnailOverallSize.x + m_spacing));
    int rowPos = (int) ((pt.y + startY * ppuY) / (m_thumbnailOverallSize.y + m_spacing));

    int itemN = (rowPos * perRow + colPos);
    if (itemN >= GetCount())
        return false;

    wxRect rect;
    GetItemRect(itemN, rect);
    if (rect.Contains(pt)) {
        n = itemN;
        return true;
    }

    return false;
}

void wxThumbnailCtrl::OnSelectAll(wxCommandEvent & WXUNUSED(event)) {
    SelectAll();
}

void wxThumbnailCtrl::OnUpdateSelectAll(wxUpdateUIEvent &event) {
    event.Enable(GetCount() > 0);
}

/// Paint the background
void wxThumbnailCtrl::PaintBackground(wxDC &dc) {
    wxColour backgroundColour = GetBackgroundColour();
    if (!backgroundColour.Ok())
        backgroundColour = wxSystemSettings::GetColour(wxSYS_COLOUR_3DFACE);

    // Clear the background
    dc.SetBrush(wxBrush(backgroundColour));
    dc.SetPen(*wxTRANSPARENT_PEN);
    wxRect windowRect(wxPoint(0, 0), GetClientSize());
    windowRect.x -= 2;
    windowRect.y -= 2;
    windowRect.width += 4;
    windowRect.height += 4;

    // We need to shift the rectangle to take into account
    // scrolling. Converting device to logical coordinates.
    CalcUnscrolledPosition(windowRect.x, windowRect.y, &windowRect.x, &windowRect.y);
    dc.DrawRectangle(windowRect);
}

void wxThumbnailCtrl::SetMouseHoverItem(int n, int flags) {
    if (m_hoverItem != n) {
        wxThumbnailEvent cmdEvent(
                wxEVT_COMMAND_THUMBNAIL_ITEM_HOVER_CHANGED,
                GetId());
        cmdEvent.SetEventObject(this);
        cmdEvent.SetFlags(flags);
        cmdEvent.SetIndex(m_hoverItem);
        m_hoverItem = n;
        GetEventHandler()->ProcessEvent(cmdEvent);
    }
}

/*!
 * wxThumbnailItem
 */

IMPLEMENT_CLASS(wxThumbnailItem, wxObject)

/// Refresh Item
bool wxThumbnailItem::Refresh(wxThumbnailCtrl *ctrl, int index) {
    wxRect r;
    ctrl->GetItemRect(index, r);
    ctrl->RefreshRect(r);
    return true;
}

/// Draw the item background. It has the default implementation.
/// You have to ovveride this function inorder to provide your implementaion.
bool
wxThumbnailItem::DrawBackground(wxDC &dc, wxThumbnailCtrl *ctrl, const wxRect &rect, const wxRect &imageRect, int style,
                                int WXUNUSED(index)) {
    auto mediumGrey = ctrl->GetUnselectedThumbnailBackgroundColour();
    auto unfocussedDarkGrey = ctrl->GetSelectedThumbnailUnfocussedBackgroundColour();
    auto focussedDarkGrey = ctrl->GetSelectedThumbnailFocussedBackgroundColour();
    wxColour darkGrey;
    if (style & wxTHUMBNAIL_FOCUSSED)
        darkGrey = focussedDarkGrey;
    else
        darkGrey = unfocussedDarkGrey;

    if (style & wxTHUMBNAIL_SELECTED) {
        wxBrush brush(darkGrey);
        wxPen pen(darkGrey);
        dc.SetBrush(brush);
        dc.SetPen(pen);
    } else {
        wxBrush brush(mediumGrey);
        wxPen pen(mediumGrey);
        dc.SetBrush(brush);
        dc.SetPen(pen);
    }

    dc.DrawRectangle(rect);

    if (style & wxTHUMBNAIL_TAGGED) {
        wxPen bluePen = ctrl->GetTagColour();
        dc.SetPen(bluePen);

        dc.DrawLine(rect.GetRight(), rect.GetTop(), rect.GetRight(), rect.GetBottom());
        dc.DrawLine(rect.GetLeft(), rect.GetBottom(), rect.GetRight() + 1, rect.GetBottom());

        dc.DrawLine(rect.GetLeft(), rect.GetTop(), rect.GetRight(), rect.GetTop());
        dc.DrawLine(rect.GetLeft(), rect.GetTop(), rect.GetLeft(), rect.GetBottom());
    } else if (ctrl->IsOutlinesShown()) {
        if (style & wxTHUMBNAIL_SELECTED) {
            dc.SetPen(*wxWHITE_PEN);
            dc.DrawLine(rect.GetRight(), rect.GetTop(), rect.GetRight(), rect.GetBottom());
            dc.DrawLine(rect.GetLeft(), rect.GetBottom(), rect.GetRight() + 1, rect.GetBottom());

            dc.SetPen(*wxBLACK_PEN);
            dc.DrawLine(rect.GetLeft(), rect.GetTop(), rect.GetRight(), rect.GetTop());
            dc.DrawLine(rect.GetLeft(), rect.GetTop(), rect.GetLeft(), rect.GetBottom());
        } else {
            dc.SetPen(*wxBLACK_PEN);
            dc.DrawLine(rect.GetRight(), rect.GetTop(), rect.GetRight(), rect.GetBottom());
            dc.DrawLine(rect.GetLeft(), rect.GetBottom(), rect.GetRight() + 1, rect.GetBottom());

            dc.SetPen(*wxWHITE_PEN);
            dc.DrawLine(rect.GetLeft(), rect.GetTop(), rect.GetRight(), rect.GetTop());
            dc.DrawLine(rect.GetLeft(), rect.GetTop(), rect.GetLeft(), rect.GetBottom());
        }
    }

    if (!m_label.IsEmpty() && (ctrl->GetWindowStyle() & wxTH_TEXT_LABEL)) {
        dc.SetFont(ctrl->GetFont());
        if (style & wxTHUMBNAIL_SELECTED)
            dc.SetTextForeground(*wxWHITE);
        else
            dc.SetTextForeground(*wxBLACK);
        dc.SetBackgroundMode(wxTRANSPARENT);

        int margin = ctrl->GetThumbnailMargin();

        wxRect fRect;
        fRect.x = rect.x + margin;
        fRect.y = rect.y + margin + imageRect.height + margin;
        fRect.width = rect.width - 2 * margin;
        fRect.height = rect.height - margin - imageRect.height - margin - margin;

        wxCoord textW, textH;
        dc.GetTextExtent(m_label, &textW, &textH);

        dc.SetClippingRegion(fRect);
        int x = fRect.x + wxMax(0, (fRect.width - textW) / 2);
        int y = fRect.y;
        dc.DrawText(m_label, x, y);
        dc.DestroyClippingRegion();
    }
    // Draw tag bitmap
    if (style & wxTHUMBNAIL_TAGGED) {
        const wxBitmap &tagBitmap = ctrl->GetTagBitmap();
        if (tagBitmap.Ok()) {
            int x = rect.x + rect.width - tagBitmap.GetWidth() - ctrl->GetThumbnailMargin();
            int y = rect.y + ctrl->GetThumbnailMargin();
            dc.DrawBitmap(tagBitmap, x, y, true);
        }
    }

    // If the item itself is the focus, draw a dotted
    // rectangle around it
    if (style & wxTHUMBNAIL_IS_FOCUS) {
        wxPen dottedPen(ctrl->GetFocusRectColour(), 1, wxDOT);
        dc.SetPen(dottedPen);
        dc.SetBrush(*wxTRANSPARENT_BRUSH);
        wxRect focusRect = imageRect;
        focusRect.x--;
        focusRect.y--;
        focusRect.width += 2;
        focusRect.height += 2;
        dc.DrawRectangle(focusRect);
    }

    return true;
}

/// Draw the item
bool wxThumbnailItem::Draw(wxDC &dc, wxThumbnailCtrl * WXUNUSED(ctrl), const wxRect &rect, int WXUNUSED(style),
                           int WXUNUSED(index)) {
    if (m_bitmap.Ok()) {
        dc.DrawBitmap(m_bitmap, rect.GetX(), rect.GetY());
    }
    return true;
}
